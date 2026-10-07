#Souleymane Barry
#Maël Fauquette
import random, gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, AUROC
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# 1. Configuration
# paramètres généraux de l'expérience : modèle de base, nombres de runs et de classes
MODEL_NAME = "distilroberta-base"          
NUM_RUNS = 3
SEEDS = [42, 123, 999]
NUM_LABELS = 100
MAX_LENGTH = 256  # longueur maximale pour la tokenization
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU si disponible

def cuda_cleanup():
    """Nettoie la mémoire GPU et le garbage collector.

    À appeler après la suppression d'un modèle pour libérer la VRAM.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def set_seed(seed: int):
    """Fixe toutes les graines aléatoires pour assurer la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#2. Construction du modèle LoRA
# crée un modèle DistilRoBERTa où seuls les modules LoRA sont entraînables
def make_lora_cls_model(base_model_name: str, r: int = 8, alpha: int = 16):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=NUM_LABELS
    )
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["query", "value"], 
        bias="none",
    )
    return get_peft_model(base_model, cfg)

#3. Évaluation 
@torch.no_grad()
def evaluate(model, dataloader):
    """Évalue précision, F1 macro et AUROC sur un dataloader."""
    model.eval()
    acc_m = Accuracy(task="multiclass", num_classes=NUM_LABELS).to(DEVICE)
    f1_m = F1Score(task="multiclass", num_classes=NUM_LABELS, average="macro").to(DEVICE)
    auroc_m = AUROC(task="multiclass", num_classes=NUM_LABELS).to(DEVICE)

    all_logits = []
    all_labels = []
    for batch in dataloader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids=ids, attention_mask=mask)
        all_logits.append(outputs.logits)
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)

    return {
        "acc": acc_m(preds, labels).item(),
        "f1": f1_m(preds, labels).item(),
        "auroc": auroc_m(probs, labels).item(),
    }


#4. Préparation des données
# charge LexGLUE Ledgar et effectue la tokenisation
# on utilise un sous-échantillon pour accélérer les tests
ds_ledgar = load_dataset("lex_glue", "ledgar")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok_fn(batch):
    # convertit le texte en ids avec padding et troncature
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_set = ds_ledgar["train"].shuffle(seed=42).select(range(20000)).map(tok_fn, batched=True)
test_set = ds_ledgar["test"].shuffle(seed=42).select(range(2000)).map(tok_fn, batched=True)

train_set = train_set.rename_column("label", "labels")
test_set = test_set.rename_column("label", "labels")

train_set.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_set.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

#5. Entraînement et évaluation
final_results = []
for run in range(NUM_RUNS):
    set_seed(SEEDS[run])
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (RoBERTa-base + LoRA)")
    model = make_lora_cls_model(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)  
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    res = evaluate(model, test_loader)
    final_results.append(res)
    print(f"Run {run+1}: Acc={res['acc']:.4f}, F1={res['f1']:.4f}, AUROC={res['auroc']:.4f}")
    del model
    cuda_cleanup()
    
#6. Calcul des moyennes et écart-types
print("\n" + "=" * 40 + "\nSYNTHÈSE DISTILROBERTA LORA\n" + "=" * 40)
for metric in ["acc", "f1", "auroc"]:
    vals = [r[metric] for r in final_results]
    print(f"{metric.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    save_path = f"results_distilroberta_lora.npy"
    np.save(save_path, final_results)

    
