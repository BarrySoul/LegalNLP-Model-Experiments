#Souleymane Barry
#Maël Fauquette
import random, gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics.classification import Accuracy, F1Score, AUROC
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# 1. Configuration
# définition des paramètres d'expérience et du matériel utilisé
MODEL_NAME = "t5-small"
NUM_RUNS = 3 
SEEDS = [42, 123, 999]
NUM_LABELS = 100  # nombre de classes du dataset
MAX_LENGTH = 256  # longueur maximale des entrées tokenisées
BATCH_SIZE = 16 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # utilise le GPU si disponible

def set_seed(seed):
    """Configure les graines aléatoires pour rendre les expériences reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cuda_cleanup():
    """Libère la mémoire GPU et le garbage collector entre deux runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# 2. Modèle LoRA pour T5
# on ajoute des adaptateurs LoRA à un modèle T5 afin d'entraîner peu de paramètres
# tout en conservant la capacité du modèle.
def make_lora_t5_model(base_model_name): 
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=100)
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o", "wi", "wo"], 
        bias="none",
    )
    return get_peft_model(model, cfg)

# 3. Évaluation
# calcule précision, F1 macro et AUROC sur un DataLoader.
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
# charge le dataset Ledgar puis tokenise les textes avec le tokenizer T5
# on réduit la taille pour des expériences rapides
ds_ledgar = load_dataset("lex_glue", "ledgar")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tok_fn(x): return tokenizer(x["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_set = ds_ledgar["train"].shuffle(seed=42).select(range(20000)).map(tok_fn, batched=True)
test_set = ds_ledgar["test"].shuffle(seed=42).select(range(2000)).map(tok_fn, batched=True)

train_set.set_format("torch", ["input_ids", "attention_mask", "label"])
test_set.set_format("torch", ["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_set.rename_column("label", "labels"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set.rename_column("label", "labels"), batch_size=BATCH_SIZE)

# 5. Entraînement et évaluation
# on répète plusieurs runs avec des graines différentes pour mesurer la variance
final_results = []
for run in range(NUM_RUNS):
    set_seed(SEEDS[run])
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} ( T5 + LoRA)")
    
    model = make_lora_t5_model(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4) 
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            # précision mixte pour accélérer l'entraînement
            with torch.amp.autocast('cuda'):
                outputs = model(ids, mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    res = evaluate(model, test_loader)
    final_results.append(res)
    print(f"Run {run+1}: Acc={res['acc']:.4f}, F1={res['f1']:.4f}, AUROC={res['auroc']:.4f}")
    # nettoyage de la mémoire GPU
    del model; cuda_cleanup()

# 6. Calcul des moyennes et écart-types
# affiche une synthèse des performances et sauvegarde les résultats
print("\n" + "="*40 + "\nSYNTHÈSE T5 LORA\n" + "="*40)
for m in ["acc", "f1", "auroc"]:
    vals = [r[m] for r in final_results]
    print(f"{m.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    save_path = f"results_t5_lora.npy"
    np.save(save_path, final_results)
