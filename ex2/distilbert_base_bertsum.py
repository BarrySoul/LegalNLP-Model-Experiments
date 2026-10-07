#Souleymane Barry
#Maël Fauquette
import torch
import numpy as np
import re, gc
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from metrics import compute_metrics
from tqdm import tqdm

# 1. Configuration
# paramètres généraux : modèle, nombre de runs et limites de longueur
MODEL_NAME = "distilbert-base-uncased"  
NUM_RUNS = 3
SEEDS = [42, 123, 999]
MAX_SENT_LEN = 64  # tokens par phrase
MAX_SENTS = 6    # nombre maximal de phrases par document
BATCH_SIZE = 4     # petit pour tenir en VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    """Fixe les graines pour rendre les expériences reproductibles."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def cuda_cleanup():
    """Vide le cache CUDA et déclenche le garbage collector."""
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def sent_tokenize(text):
    """Découpe un texte en phrases en utilisant ponctuation de fin."""
    return re.split(r'(?<=[.!?]) +', text)

# 2. Modèle extractif basé sur DistilBERT
class DistilBertSumExtractive(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # encodeur DistilBERT
        self.distilbert = AutoModel.from_pretrained(model_name) 
        # classificateur linéaire sur l'embedding [CLS]
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # input_ids shape: (batch, sents, sent_len)
        B, S, L = input_ids.shape  
        # condenser dimension phrases dans le batch
        input_ids = input_ids.view(B * S, L)  
        attention_mask = attention_mask.view(B * S, L)  
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
        scores = self.sigmoid(self.classifier(cls_embeddings))  # probabilité d'inclusion
        return scores.view(B, S) 

# 3. Préparation des données
# on utilise le jeu 'billsum' avec un split CA puis on crée train/test
# chaque document est découpé en phrases puis tokenisé individuellement
ds_billsum = load_dataset("FiscalNote/billsum", split="ca_test").train_test_split(test_size=0.2, seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_fn(examples):
    all_input_ids, all_masks = [], []
    for text in examples["text"]:
        # limite à MAX_SENTS phrases
        sentences = sent_tokenize(text)[:MAX_SENTS]
        if len(sentences) < MAX_SENTS:
            sentences += [""] * (MAX_SENTS - len(sentences))
        
        enc = tokenizer(sentences, truncation=True, padding="max_length", 
                        max_length=MAX_SENT_LEN, return_tensors="pt")
        all_input_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
    
    return {"input_ids": torch.stack(all_input_ids), "attention_mask": torch.stack(all_masks), "summary": examples["summary"]}

tokenized_ds = ds_billsum.map(preprocess_fn, batched=True, remove_columns=["text"])
tokenized_ds.set_format("torch")

# 4. Évaluation
@torch.no_grad()
def evaluate(model, dataloader, original_ds):
    """Construit des résumés extractifs en sélectionnant les 3 phrases
dans chaque document avec les scores les plus élevés."""
    model.eval()
    preds_all, refs_all = [], []
    
    idx = 0
    for batch in dataloader:
        ids, mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
        scores = model(ids, mask) 
        
        for i in range(scores.shape[0]):
            top_indices = torch.argsort(scores[i], descending=True)[:3].cpu().numpy()
            sentences = sent_tokenize(original_ds[idx]["text"])
            pred_summary = " ".join([sentences[j] for j in top_indices if j < len(sentences)])
            preds_all.append(pred_summary)
            refs_all.append(original_ds[idx]["summary"])
            idx += 1
            if idx >= 50: break 
        if idx >= 50: break

    return compute_metrics(preds_all, refs_all)

# 5. Exécution
# boucle d'entraînement et évaluation multi-run
final_results = []
for run in range(NUM_RUNS):
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (Seed: {SEEDS[run]})")
    set_seed(SEEDS[run])
    
    model = DistilBertSumExtractive(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # utilisation d'un petit sous-ensemble pour la vitesse
    train_loader = DataLoader(tokenized_ds["train"].select(range(500)), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids, mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(ids, mask)
                # perte MSE contre vecteur de 1 (simulateur de supervision)
                loss = nn.functional.mse_loss(logits, torch.ones_like(logits))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

    test_loader = DataLoader(tokenized_ds["test"], batch_size=BATCH_SIZE)
    metrics = evaluate(model, test_loader, ds_billsum["test"])
    final_results.append(metrics)
    print(f"Run {run+1}: R1={metrics['rouge1_f1']:.4f}, BLEU={metrics['bleu']:.4f}")
    del model; cuda_cleanup()

for m in ["rouge1_f1", "rouge2_f1", "bleu"]:
    vals = [r[m] for r in final_results]
    print(f"FINAL {m.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")