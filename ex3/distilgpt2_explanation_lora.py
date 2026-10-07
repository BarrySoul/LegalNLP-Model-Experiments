# Souleymane Barry
# Maël Fauquette
import json, os, random, gc, re
from typing import Dict, List
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# 1. Configuration LoRA et Entraînement
MODEL_NAME = "distilgpt2" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
SEED = 42
TRAIN_MAX_EX = 8000 
EVAL_N = 30  
MAX_SEQ_LEN = 384  
BATCH_SIZE = 2  
GRAD_ACCUM_STEPS = 8   
LR = 2e-4
EPOCHS = 1
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

OUT_DIR = "distilgpt2_lora_exo3"
OUTPUT_JSONL = os.path.join(OUT_DIR, "distilgpt2_lora_outputs.jsonl")

@dataclass
class DecodingConfig:
    name: str
    do_sample: bool
    temperature: float
    top_p: float
    num_beams: int
    repetition_penalty: float
    no_repeat_ngram_size: int

LORA_DECODING = DecodingConfig(
    name="lora_default",
    do_sample=False,
    temperature=1.0, 
    top_p=1.0,
    num_beams=2,
    repetition_penalty=1.10,
    no_repeat_ngram_size=3,
)

# 2. Heuristiques de Risque (Safety)
_RE_NEW_OBLIG = re.compile(r"\b(must|shall|required to|is required to|obligated to|may not)\b", re.IGNORECASE)
_RE_ADVICE = re.compile(r"\b(you should|we recommend|consult (a|your) (lawyer|attorney)|seek legal advice)\b", re.IGNORECASE)
_RE_NEW_PARTIES = re.compile(r"\b(third party|licensor|licensee|affiliate|subsidiary)\b", re.IGNORECASE)
_RE_DEADLINES = re.compile(r"\b(within \d+|no later than|deadline|days? after)\b", re.IGNORECASE)

def risk_flags(expl: str) -> Dict[str, bool]:
    """Renvoie un dictionnaire de drapeaux indiquant certains risques.

    Ces heuristiques détectent obligations, conseils juridiques, parties,
    délais, ou longueur inappropriée de l'explication.
    """
    e = expl or ""
    return {
        "mentions_new_obligations": bool(_RE_NEW_OBLIG.search(e)),
        "gives_legal_advice": bool(_RE_ADVICE.search(e)),
        "mentions_parties_terms": bool(_RE_NEW_PARTIES.search(e)),
        "mentions_deadlines": bool(_RE_DEADLINES.search(e)),
        "empty_or_too_short": len(e.split()) < 8,
        "too_long": len(e.split()) > 85,
    }

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def normalize_ws(text: str) -> str:
    return " ".join(text.split())

def build_prompt(clause: str) -> str:
    """Construit le prompt d'entrée donné un texte de clause.

    Le prompt inclut des instructions claires pour l'assistant fictif et
    rappelle de ne pas ajouter d'obligations ou de conseils.
    """
    clause = normalize_ws(clause)
    return (
        "You are a careful educational legal assistant.\n"
        "Task: Explain the clause in plain English for a non-lawyer.\n"
        "Rules:\n"
        "- Be faithful to the clause.\n"
        "- Do NOT add new obligations, exceptions, deadlines, penalties, or parties.\n"
        "- Do NOT give legal advice.\n"
        "- If the clause is unclear, say so briefly.\n"
        "- Output 2 to 4 short sentences.\n\n"
        f"Clause:\n{clause}\n\n"
        "Explanation:\n"
    )
# 3. Préparation des données (Synthetic Target)
def synthetic_target_from_label(label_name: str) -> str:
    """Génère un texte d'entraînement synthétique basé sur le nom de label.

    Utile pour produire des exemples d'explication lorsque la clause est
    étiquetée par catégorie.
    """
    label_name = label_name.replace("_", " ").strip().lower()
    return (
        f"This clause is about {label_name}. "
        "It describes the general terms and conditions related to this topic in the agreement. "
        "It should be read together with the rest of the contract for full context."
    )

def make_training_example(tokenizer, clause: str, label_name: str) -> Dict[str, torch.Tensor]:
    """Confectionne un exemple d'entraînement LoRA à partir d'une clause.

    Le prompt est construit, concaténé à une explication synthétique, puis
    tokenisé. Les tokens du prompt et du padding sont masqués (-100).
    """
    prompt = build_prompt(clause)
    target = synthetic_target_from_label(label_name)
    full = prompt + target + tokenizer.eos_token
    enc = tokenizer(full, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    prompt_enc = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]
    labels = input_ids.clone()
    prompt_len = int(prompt_enc["attention_mask"][0].sum().item())
    labels[:prompt_len] = -100
    labels[attn == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }

@torch.no_grad()
def generate_explanation(model, tok, clause: str) -> tuple[str, str]:
    prompt = build_prompt(clause)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False).to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=96,
        do_sample=LORA_DECODING.do_sample,
        num_beams=LORA_DECODING.num_beams,
        repetition_penalty=LORA_DECODING.repetition_penalty,
        no_repeat_ngram_size=LORA_DECODING.no_repeat_ngram_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        early_stopping=True,
    )
    gen_ids = out[0, enc["input_ids"].shape[1]:]
    return normalize_ws(tok.decode(gen_ids, skip_special_tokens=True).strip()), prompt

# 4. Prétraitement pour l'entraînement : création de prompts et masquage des tokens de la partie "prompt" dans les labels
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    cuda_cleanup()

    ds = load_dataset("lex_glue", "ledgar")
    train = ds["train"].shuffle(seed=SEED).select(range(min(TRAIN_MAX_EX, len(ds["train"]))))
    test = ds["test"]
    label_names = ds["train"].features["label"].names

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32).to(DEVICE)
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM, target_modules=["c_attn", "c_proj"])
    model = get_peft_model(base_model, lora_cfg)
    
    train_items = [make_training_example(tok, ex["text"], label_names[int(ex["label"])]) for ex in train]
    loader = DataLoader(train_items, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(optim, int(WARMUP_RATIO * (EPOCHS * len(loader)//GRAD_ACCUM_STEPS)), EPOCHS * len(loader)//GRAD_ACCUM_STEPS)
    scaler = torch.amp.GradScaler("cuda")
    
    model.train()
    for epoch in range(EPOCHS):
        optim.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for it, batch in enumerate(pbar): 
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                raw_loss = model(**batch).loss
                scaled_loss = raw_loss / GRAD_ACCUM_STEPS
            scaler.scale(scaled_loss).backward()
            pbar.set_postfix({"loss": f"{raw_loss.item():.4f}"})
            if (it + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                sched.step()

    model.eval()
    subset = test.shuffle(seed=SEED).select(range(min(EVAL_N, len(test))))
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for i, ex in enumerate(subset):
            clause = ex["text"]
            expl, prompt = generate_explanation(model, tok, clause)
            row = {
                "i": i,
                "model": MODEL_NAME,
                "adaptation": "LoRA fine-tuning",
                "prompting": "instruction prompt",
                "decoding": asdict(LORA_DECODING),
                "max_input_tokens": 256,
                "max_new_tokens": 96,
                "clause": clause,
                "prompt": prompt,
                "explanation": expl,
                "risk_flags": risk_flags(expl),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
