#Souleymane Barry
#Maël Fauquette
import random, gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics.classification import Accuracy, F1Score, AUROC
from tqdm import tqdm

# 1. Configuration
# paramètres de l'expérience, modèle et dispositif
MODEL_NAME = "distilgpt2"
NUM_RUNS = 3 
SEEDS = [42, 123, 999]
MAX_LENGTH = 256  # longueur maximale des séquences
BATCH_SIZE = 16 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 100

def set_seed(seed):
    """Fixe les graines pour la reproductibilité (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cuda_cleanup():
    """Nettoie la mémoire GPU entre deux runs pour éviter les fuites."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 2. Modèle Decoder-only avec Linear Probing
# Cette fonction retourne un modèle GPT2 où tous les paramètres sont gelés
# sauf ceux liés à la tête de scoring ('score' layer) afin de n'entraîner qu'un
# petit nombre de paramètres.
def make_gpt_probe_model(base_model_name):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 n'avait pas de pad_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=NUM_LABELS
    )

    # s'assurer que le token de padding correspond à eos
    model.config.pad_token_id = model.config.eos_token_id
    for name, param in model.named_parameters():
        # ne dégel que les poids de la couche de classification linéaire
        if "score" not in name: 
            param.requires_grad = False
            
    return model, tokenizer

# 3. Évaluation
# Calcule des métriques standards (accuracy, F1 macro, AUROC) sur un DataLoader
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    acc_m = Accuracy(task="multiclass", num_classes=NUM_LABELS).to(DEVICE)
    f1_m = F1Score(task="multiclass", num_classes=NUM_LABELS, average="macro").to(DEVICE)
    auroc_m = AUROC(task="multiclass", num_classes=NUM_LABELS).to(DEVICE)
    
    all_logits, all_labels = [], []
    for batch in dataloader:
        ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
        outputs = model(ids, attention_mask=mask)
        all_logits.append(outputs.logits)
        all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)

    return {
        "acc": acc_m(preds, labels).item(),
        "f1": f1_m(preds, labels).item(),
        "auroc": auroc_m(probs, labels).item()
    }

# 4. Chargement et Préparation des données
# on charge le jeu de données LexGLUE - Ledgar et on construit le tokenizer
# via la fonction de création de modèle pour s'assurer d'avoir le même objet
# tokenizer utilisé lors de l'entraînement.
ds_ledgar = load_dataset("lex_glue", "ledgar")
model_tmp, tokenizer = make_gpt_probe_model(MODEL_NAME)

def tok_fn(x): 
    # tokenisation avec troncature et padding fixe
    return tokenizer(x["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

# sous-ensembles pour accélérer l'entraînement/test
train_set = ds_ledgar["train"].shuffle(seed=42).select(range(20000)).map(tok_fn, batched=True)
test_set = ds_ledgar["test"].shuffle(seed=42).select(range(2000)).map(tok_fn, batched=True)

# mise en forme pour PyTorch
train_set.set_format("torch", ["input_ids", "attention_mask", "label"])
test_set.set_format("torch", ["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_set.rename_column("label", "labels"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set.rename_column("label", "labels"), batch_size=BATCH_SIZE)

# 5. Entraînement et évaluation
# multiple runs avec graines différentes pour estimer la variance
final_results = []
for run in range(NUM_RUNS):
    set_seed(SEEDS[run])
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (DistilGPT2 Linear Probing)")
    
    model, _ = make_gpt_probe_model(MODEL_NAME)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.score.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(3):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")

        for batch in loop:
            optimizer.zero_grad()
            ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            
            # mixte précision pour accélérer sur GPU
            with torch.amp.autocast('cuda'):
                outputs = model(ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    res = evaluate(model, test_loader)
    final_results.append(res)
    print(f"Run {run+1}: Acc={res['acc']:.4f}, F1={res['f1']:.4f}, AUROC={res['auroc']:.4f}")
    # libère le modèle et la mémoire GPU après chaque run
    del model; cuda_cleanup()

# 6. Calcul des moyennes et écart-types
# synthèse des résultats sur tous les runs
print("\n" + "="*40 + "\nSYNTHÈSE DISTILGPT2 PROBE\n" + "="*40)
for m in ["acc", "f1", "auroc"]:
    vals = [r[m] for r in final_results]
    print(f"{m.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

