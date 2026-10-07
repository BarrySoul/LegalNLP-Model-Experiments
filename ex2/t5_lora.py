#Souleymane Barry
#Maël Fauquette
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from metrics import compute_metrics
from tqdm import tqdm

# 1. Configuration
# paramètres généraux de l'expérience et dimensions des séquences
MODEL_NAME = "t5-small"
NUM_RUNS = 3
SEEDS = [42,123,999]
MAX_INPUT_LEN = 512  # tokens source maximum
MAX_TARGET_LEN = 128  # tokens cible maximum
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU si disponible

def set_seed(seed):
    """Fixe les graines pour Python et NumPy pour assurer la reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)

# 2. Chargement de BillSum
# on récupère un split canadien puis on sépare en train/test
# le tokenizer T5 sera utilisé pour encoder les textes et les résumés
ds_billsum = load_dataset("FiscalNote/billsum", split="ca_test").train_test_split(test_size=0.2, seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_fn(examples):
    """Prétraite textes et résumés : ajoute le préfixe T5 et prépare les labels.

    Les labels padding sont remplacés par -100 pour ignorer la perte.
    """
    inputs = ["summarize: " + doc for doc in examples["text"]]
    enc = tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True, padding="max_length")
    dec = tokenizer(
        text_target=examples["summary"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    )
    enc["labels"] = [
        [(id_ if id_ != tokenizer.pad_token_id else -100) for id_ in label]
        for label in dec["input_ids"]
    ]
    return enc


tokenized_ds = ds_billsum.map(
    preprocess_fn,
    batched=True,
    remove_columns=ds_billsum["train"].column_names  
)

# 3. Évaluation
@torch.no_grad()
def evaluate(model, dataloader):
    """Génère des sorties pour un DataLoader et calcule les métriques."""
    model.eval()
    preds_all, refs_all = [], []

    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        gen_ids = model.generate(
            input_ids=batch["input_ids"],
            max_new_tokens=MAX_TARGET_LEN,
            num_beams=2,
            do_sample=False,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        labels = batch["labels"].detach().cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds_all.extend(preds)
        refs_all.extend(refs)

    return compute_metrics(preds_all, refs_all)  
    

# 4. Boucle des runs
# chaque run utilise une graine différente pour mesurer la variance
final_results = []
for run in range(NUM_RUNS):
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (Seed: {SEEDS[run]})") 
    set_seed(SEEDS[run]) 
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = base_model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    scaler = torch.amp.GradScaler('cuda') 
    
    train_loader = DataLoader(tokenized_ds["train"], batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model))
    model.train()
    for epoch in range(3): 
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast('cuda'): 
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    test_loader = DataLoader(tokenized_ds["test"], batch_size=BATCH_SIZE, 
                             collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model))
    metrics = evaluate(model, test_loader)
    final_results.append(metrics)
    print(f"Resultats Run {run+1}: R1={metrics['rouge1_f1']:.4f},R2={metrics['rouge2_f1']:.4f}, BLEU={metrics['bleu']:.4f}")

# 4. Calcul des statistiques 
print("\n" + "="*30 + "\nBILAN FINAL\n" + "="*30)
for m_name in ["rouge1_f1", "rouge2_f1", "bleu"]:
    vals = [r[m_name] for r in final_results]
    print(f"{m_name.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
