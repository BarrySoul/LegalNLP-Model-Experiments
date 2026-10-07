#Souleymane Barry
#Maël Fauquette
import torch
import numpy as np
import random
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torchmetrics.classification import Accuracy,F1Score, AUROC
from tqdm import tqdm

# 1. Configuration et reproductibilité
# Définit le modèle de base et les paramètres de l'expérience
MODEL_NAME = "distilbert-base-uncased" 
NUM_RUNS = 3 
SEEDS = [42, 123, 999]
MAX_LENGTH = 256  # longueur maximale des séquences tokenisées
BATCH_SIZE = 16  # taille de lot pour l'entraînement
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # utilise GPU si disponible

ds_ledgar = load_dataset("lex_glue", "ledgar")
label_names = ds_ledgar["train"].features["label"].names
NUM_LABELS = len(label_names) 

def set_seed(seed):
    """Configure les graines pour assurer la reproductibilité.

    Les graines sont appliquées à Python, NumPy et PyTorch (CPU et GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 2. Modèle Linear Probe pour 100 classes
# Le BERT pré-entraîné est gelé et seul un classificateur linéaire est entraîné
class LegalClassifierBERT(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # charge le modèle de base et désactive la mise à jour des poids
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        # couche de classification linéaire appliquée sur le token [CLS]
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # propagation avant : encoder puis classificateur
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        return self.classifier(self.dropout(pooled_output))

# 3. Fonctions d'entraînement et d'évaluation
def evaluate(model, dataloader):
    """Calcul des métriques sur un ensemble de données donné.

    Retourne un dictionnaire avec l'exactitude, le F1 macro et l'AUROC.
    """
    model.eval()
    acc_metric = Accuracy(task="multiclass", num_classes=NUM_LABELS).to(DEVICE)
    f1_metric = F1Score(task="multiclass", num_classes=NUM_LABELS, average="macro").to(DEVICE)
    auroc_metric = AUROC(task="multiclass", num_classes=NUM_LABELS).to(DEVICE)
    
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            logits = model(ids, mask)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    probs = torch.softmax(all_logits, dim=1)
    preds = torch.argmax(all_logits, dim=1)

    return {
        "acc": acc_metric(preds, all_labels).item(),
        "f1": f1_metric(preds, all_labels).item(),
        "auroc": auroc_metric(probs, all_labels).item()
    }

# 4. Préparation des données 
# initialise le tokenizer et sélectionne des sous-ensembles de données pour accélérer l'entraînement
# sur du matériel limité
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_sub = ds_ledgar["train"].shuffle(seed=42).select(range(20000))
test_sub = ds_ledgar["test"].shuffle(seed=42).select(range(2000))

def tokenize_fn(ex):
    """Tokenise un exemple en appliquant troncature et padding.
    Cette fonction sera utilisée avec `map` pour prétraiter tout le dataset.
    """
    return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_train = train_sub.map(tokenize_fn, batched=True).rename_column("label", "labels")
tokenized_test = test_sub.map(tokenize_fn, batched=True).rename_column("label", "labels")
tokenized_train.set_format("torch", ["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", ["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE)

# 5. Entraînement et évaluation
# Boucle sur plusieurs runs pour calculer la moyenne des performances
final_results = []
for run in range(NUM_RUNS):
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (Seed: {SEEDS[run]})")
    set_seed(SEEDS[run])
    
    # création du modèle et de l'optimiseur (seuls les poids du classificateur sont mis à jour)
    model = LegalClassifierBERT(MODEL_NAME, NUM_LABELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') 

    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            
            # utilisation mixte précision pour accélérer l'entraînement sur GPU
            with torch.amp.autocast('cuda'):
                logits = model(ids, mask)
                loss = loss_fn(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    metrics = evaluate(model, test_loader)
    final_results.append(metrics)
    print(f"Resultats Run {run+1}: Acc={metrics['acc']:.4f}, F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}")

# 6. Calcul des moyennes et écart-types
# Affiche les performances moyennes et la variation entre les runs
for m_name in ["acc", "f1", "auroc"]:
    values = [r[m_name] for r in final_results]
    print(f"FINAL {m_name.upper()}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
