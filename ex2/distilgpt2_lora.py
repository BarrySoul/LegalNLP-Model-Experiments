#Souleymane Barry
#Maël Fauquette
import torch
import numpy as np
import gc
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import compute_metrics
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# 1. Configuration
# paramètres du modèle et de l'expérience
MODEL_NAME = "distilgpt2"
NUM_RUNS = 3
SEEDS = [42, 123, 999]
MAX_SEQ_LEN = 384  # longueur maximale totale des séquences
ANSWER_MARGIN = 128  # tokens réservés à la génération
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # utilise GPU si possible

def set_seed(seed):
    """Fixe les graines pour la reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def cuda_cleanup():
    """Nettoie la mémoire GPU et déclenche le garbage collector."""
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# 2. Préparation des données
# jeu de données Billsum, prompts pour un modèle causal et génération de labels masqués
ds_billsum = load_dataset("FiscalNote/billsum", split="ca_test").train_test_split(test_size=0.2, seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 n'a pas de padding token

def preprocess_fn(examples):
    """Construit les entrées et labels pour le fine-tuning causal.

    Les prompts et cibles sont concaténées; les tokens correspondant
    au prompt sont masqués (-100) pour le calcul de la perte.
    """
    prompts = [f"Summarize this legal text:\n{t}\nSummary:" for t in examples["text"]]
    targets = [f" {s}{tokenizer.eos_token}" for s in examples["summary"]]
    
    inputs = tokenizer(prompts, truncation=True, max_length=MAX_SEQ_LEN - ANSWER_MARGIN)
    full_enc = tokenizer([p + t for p, t in zip(prompts, targets)], 
                         truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    
    labels = []
    for i in range(len(full_enc["input_ids"])):
        label = list(full_enc["input_ids"][i])
        prompt_len = len(inputs["input_ids"][i])
        for j in range(prompt_len): label[j] = -100
        for j in range(len(label)):
            if full_enc["input_ids"][i][j] == tokenizer.pad_token_id: label[j] = -100
        labels.append(label)
        
    full_enc["labels"] = labels
    return full_enc

tokenized_ds = ds_billsum.map(preprocess_fn, batched=True, remove_columns=ds_billsum["train"].column_names)
tokenized_ds.set_format("torch")

# 3. Évaluation
@torch.no_grad()
def evaluate(model, dataloader):
    """Génère des résumés à partir des prompts et compare aux références."""
    model.eval()
    preds_all, refs_all = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"]
        
        for i in range(input_ids.shape[0]):
            # longueur du prompt masqué
            prompt_end = (labels[i] == -100).sum().item()
            
            gen_ids = model.generate(
                input_ids[i:i+1, :prompt_end],
                max_new_tokens=ANSWER_MARGIN,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=2
            )
            
            pred = tokenizer.decode(gen_ids[0][prompt_end:], skip_special_tokens=True)
            ref_ids = labels[i][labels[i] != -100]
            ref = tokenizer.decode(ref_ids, skip_special_tokens=True)
            
            preds_all.append(pred)
            refs_all.append(ref)
        if len(preds_all) >= 50: break

    return compute_metrics(preds_all, refs_all)

# 4. Exécution
final_results = []
for run in range(NUM_RUNS):
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (Seed: {SEEDS[run]})")
    set_seed(SEEDS[run])

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, target_modules=["c_attn"])
    model = get_peft_model(model, peft_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler('cuda')
    train_loader = DataLoader(tokenized_ds["train"], batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast('cuda'):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

    test_loader = DataLoader(tokenized_ds["test"], batch_size=BATCH_SIZE)
    metrics = evaluate(model, test_loader)
    final_results.append(metrics)
    print(f"Run {run+1}: R1={metrics['rouge1_f1']:.4f}, BLEU={metrics['bleu']:.4f}")
    del model; cuda_cleanup()

for m in ["rouge1_f1", "rouge2_f1", "bleu"]:
    vals = [r[m] for r in final_results]
    print(f"FINAL {m.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")