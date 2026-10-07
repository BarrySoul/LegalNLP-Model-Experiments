#Souleymane Barry
#Maël Fauquette
import torch
import numpy as np
import gc
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from metrics import compute_metrics
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# 1. Configuration
# paramètres du modèle et de l'expérience
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # nom du modèle TinyLlama
NUM_RUNS = 3
SEEDS = [42, 123, 999]  # graines pour répéter l'entraînement
MAX_SEQ_LEN = 256  # longueur max de la séquence (prompt+réponse)
ANSWER_MARGIN = 128  # tokens réservés pour la génération
BATCH_SIZE = 2  # très petit à cause de la taille du modèle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    """Fixe les graines pour Python/NumPy et PyTorch GPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def cuda_cleanup():
    """Nettoie le cache CUDA et déclenche le GC pour libérer la VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 2. Préparation des données
# dataset Billsum, split canadien, puis tokenization
ds_billsum = load_dataset("FiscalNote/billsum", split="ca_test").train_test_split(test_size=0.2, seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # définir pad_token pour TinyLlama

def preprocess_fn(examples):
    """Prépare les entrées et labels pour fine-tuning causal.

    - prompts et cibles concaténées
    - les tokens du prompt et du padding sont masqués (-100)
    """
    prompts = [f"Summarize this legal text:\n{t}\nSummary:" for t in examples["text"]]
    targets = [f" {s}{tokenizer.eos_token}" for s in examples["summary"]]
    
    # Encodage séparé pour connaître la longueur du prompt
    prompt_encs = tokenizer(prompts, truncation=True, max_length=MAX_SEQ_LEN - ANSWER_MARGIN)
    # Encodage complet (prompt + cible)
    full_encs = tokenizer([p + t for p, t in zip(prompts, targets)], 
                          truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    
    labels_all = []
    for i in range(len(full_encs["input_ids"])):
        labels = list(full_encs["input_ids"][i])
        prompt_len = len(prompt_encs["input_ids"][i])
        
        # Masquer le prompt et le padding dans les labels (-100)
        for j in range(prompt_len):
            labels[j] = -100
        for j in range(len(labels)):
            if full_encs["input_ids"][i][j] == tokenizer.pad_token_id:
                labels[j] = -100
        labels_all.append(labels)
    
    full_encs["labels"] = labels_all
    return full_encs

tokenized_ds = ds_billsum.map(preprocess_fn, batched=True, remove_columns=ds_billsum["train"].column_names)
tokenized_ds.set_format("torch")

# 3. Évaluation (Modifiée pour gérer proprement le DataLoader)
@torch.no_grad()
def evaluate(model, dataloader):
    """Génère des résumés à partir des prompts et compare aux références.

    On limite à ~50 exemples pour accélérer l'évaluation.
    """
    model.eval()
    preds_all, refs_all = [], []
    max_eval_steps = 50 // BATCH_SIZE # arrêt après 50 exemples approximatifs

    for i, batch in enumerate(dataloader):
        if i >= max_eval_steps: break
        
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        for b in range(input_ids.shape[0]):
            # fin du prompt : nombre de -100 dans labels
            prompt_end = (labels[b] == -100).sum().item()
            
            # génération depuis le prompt seulement
            gen_ids = model.generate(
                input_ids=input_ids[b:b+1, :prompt_end],
                max_new_tokens=ANSWER_MARGIN,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=2,
            )

            # décoder la partie générée
            pred = tokenizer.decode(gen_ids[0][prompt_end:], skip_special_tokens=True)
            
            # décoder la référence (partie non masquée)
            ref_ids = labels[b][labels[b] != -100]
            ref = tokenizer.decode(ref_ids, skip_special_tokens=True)

            preds_all.append(pred)
            refs_all.append(ref)

    return compute_metrics(preds_all, refs_all)

# 4. Boucle d'exécution
# itère sur plusieurs runs pour estimer la variance due aux graines
final_results = []
for run in range(NUM_RUNS):
    print(f"\n>>> RUN {run+1}/{NUM_RUNS} (Seed: {SEEDS[run]})")
    set_seed(SEEDS[run])
    cuda_cleanup()
    
    # chargement du modèle TinyLlama en float16 pour gagner en mémoire
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, cfg)  # applique LoRA
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    train_loader = DataLoader(tokenized_ds["train"], batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(3):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # création d'un DataLoader pour évaluation (utilise DataCollator Seq2Seq)
    test_loader = DataLoader(tokenized_ds["test"], batch_size=BATCH_SIZE, 
                             collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model))
    
    metrics = evaluate(model, test_loader)
    final_results.append(metrics)
    print(f"Résultats Run {run+1}: R1={metrics['rouge1_f1']:.4f}, R2={metrics['rouge2_f1']:.4f}, BLEU={metrics['bleu']:.4f}")
    
    del model; cuda_cleanup()

# 5. Synthèse finale
print("\n" + "="*30 + "\nFINAL BENCHMARK TINYLLAMA\n" + "="*30)
for metric in ["rouge1_f1", "rouge2_f1", "bleu"]:
    vals = [r[metric] for r in final_results]
    print(f"{metric.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")