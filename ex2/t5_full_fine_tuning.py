#Souleymane Barry
#Maël Fauquette
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from metrics import compute_metrics
from tqdm import tqdm

MODEL_NAME = "t5-small"
NUM_RUNS = 3
SEEDS = [42,123,999]
MAX_INPUT_LEN = 512  # longueur maximale du texte source
MAX_TARGET_LEN = 256  # longueur maximale du résumé généré
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU si disponible

def set_seed(seed):
    """Fixe les graines Python et NumPy pour la reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)

# Charger et découper BillSum
# on récupère un split CA, puis on sépare train/test
ds_billsum = load_dataset("FiscalNote/billsum", split="ca_test").train_test_split(test_size=0.2, seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_fn(examples):
    """Prétraite les textes et résumés pour l'entraînement seq2seq.

    Le préfixe 'summarize:' est ajouté suivant les conventions T5.
    Les pads sont remplacés par -100 pour ignorer la perte.
    """
    inputs = ["summarize: " + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs,
                             max_length=MAX_INPUT_LEN,
                             truncation=True,
                             padding="max_length")
    labels = tokenizer(text_target=examples["summary"],
                       max_length=MAX_TARGET_LEN,
                       truncation=True,
                       padding="max_length")
    ids = labels["input_ids"]
    # Remplacer les tokens de padding par -100
    labels_out = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in ids]
    model_inputs["labels"] = labels_out
    return model_inputs

# Appliquer le prétraitement et supprimer les colonnes originales
tokenized_ds = ds_billsum.map(
    preprocess_fn,
    batched=True,
    remove_columns=ds_billsum["train"].column_names
)


@torch.no_grad()
def evaluate(model, dataloader):
    """Génère des sorties et calcule métriques comparées aux références."""
    model.eval()
    preds_all, refs_all = [], []

    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            max_new_tokens=MAX_TARGET_LEN,
            num_beams=2,
            do_sample=False,
        )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        labels = batch["labels"].detach().cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds_all.extend(preds)
        refs_all.extend(refs)

    return compute_metrics(preds_all, refs_all)  

final_results = []
for run in range(NUM_RUNS):
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (Seed: {SEEDS[run]})") 
    set_seed(SEEDS[run])
    
    # init modèle et optimiseur
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    scaler = torch.amp.GradScaler('cuda') 
    
    train_loader = DataLoader(tokenized_ds["train"], batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model))

    model.train()
    for epoch in range(3): 
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1} - Training"):
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
    print(f"Resultats Run {run+1}: R1={metrics['rouge1_f1']:.4f}, R2={metrics['rouge2_f1']:.4f}, BLEU={metrics['bleu']:.4f}")

for m_name in ["rouge1_f1", "rouge2_f1", "bleu"]:
    values = [r[m_name] for r in final_results]
    print(f"FINAL {m_name.upper()}: {np.mean(values):.4f} +/- {np.std(values):.4f}") 