#Souleymane Barry
#Maël Fauquette
import random, gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics.classification import Accuracy, F1Score, AUROC
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# 1. Configuration 
# paramètres de l'expérience et du modèle
MODEL_NAME = "distilbert-base-uncased"
NUM_RUNS = 3 
SEEDS = [42, 123, 999]
NUM_LABELS = 100  # nombre de classes dans le dataset
MAX_LENGTH = 256  # longueur maximale des textes tokenisés
BATCH_SIZE = 16 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # utilise GPU si disponible

def cuda_cleanup():
    """Libère la mémoire GPU entre deux runs pour éviter les fuites.

    Appelle le ramasse‑miettes et vide le cache CUDA s'il est actif.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def set_seed(seed):
    """Définit toutes les graines aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 2. Modèle LoRA
# Crée un modèle de classification où seuls des modules LoRA sont entraînés
# (low‑rank adaptation) pour réduire le nombre de paramètres mis à jour.
def make_lora_cls_model(base_model_name, r=8, alpha=16):
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=NUM_LABELS)
    
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"], 
        bias="none",
    )
    return get_peft_model(model, cfg)

# 3. Évaluation
# calcul des métriques sur un DataLoader
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    acc_m = Accuracy(task="multiclass", num_classes=NUM_LABELS).to(DEVICE)
    f1_m = F1Score(task="multiclass", num_classes=NUM_LABELS, average="macro").to(DEVICE)
    auroc_m = AUROC(task="multiclass", num_classes=NUM_LABELS).to(DEVICE)
    
    all_logits, all_labels = [], []
    for batch in dataloader:
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(ids, mask)
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

# 4. Préparation des données
# chargement du dataset et tokenization
ds_ledgar = load_dataset("lex_glue", "ledgar")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tok_fn(x): return tokenizer(x["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

# on réduit la taille des ensembles pour accélérer les tests
train_set = ds_ledgar["train"].shuffle(seed=42).select(range(20000)).map(tok_fn, batched=True)
test_set = ds_ledgar["test"].shuffle(seed=42).select(range(2000)).map(tok_fn, batched=True)

# format PyTorch pour DataLoader
train_set.set_format("torch", ["input_ids", "attention_mask", "label"])
test_set.set_format("torch", ["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_set.rename_column("label", "labels"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set.rename_column("label", "labels"), batch_size=BATCH_SIZE)

# 5. Entraînement et évaluation
# effectue plusieurs runs avec différentes graines
final_results = []
for run in range(NUM_RUNS):
    set_seed(SEEDS[run])
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (Seed: {SEEDS[run]})")
    
    model = make_lora_cls_model(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler('cuda') 

    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            
            # précision mixte pour réduire la consommation mémoire et accélérer
            with torch.amp.autocast('cuda'): 
                outputs = model(ids, mask, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    metrics = evaluate(model, test_loader)
    final_results.append(metrics)
    print(f"Resultats Run {run+1}: Acc={metrics['acc']:.4f}, F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}")
    
    # libération avant le prochain run
    del model; cuda_cleanup()

# 6. Calcul des moyennes et écart-types
# affichage des performances moyennes et de la variation entre runs
for m_name in ["acc", "f1", "auroc"]:
    values = [r[m_name] for r in final_results]
    print(f"FINAL {m_name.upper()}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    