# Souleymane Barry
# Maël Fauquette
import json, math, os, random, gc, re
from typing import Dict
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# 1. Configuration du Modèle et de l'Entraînement
# On utilise Qwen2.5-0.5B-Instruct, pour un entraînement LoRA sur GPU.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TRAIN_MAX_EX = 8000
EVAL_N = 30
MAX_SEQ_LEN = 512
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LR = 2e-4
EPOCHS = 1
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01 

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

OUT_DIR = "qwen25_05b_lora_exo3"
OUTPUT_JSONL = os.path.join(OUT_DIR, "qwen25_05b_lora_outputs.jsonl")

# 2. Fonctions Utiles : Seed, Nettoyage GPU, Prompting, Masquage, Décodage
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

# 3. Configurations de Décodage
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
    name="qwen_lora_default",
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    num_beams=1,
    repetition_penalty=1.10,
    no_repeat_ngram_size=3,
)

# Ces configurations sont utilisées à la fois pour l'entraînement (masquage) et l'évaluation (décodage contrôlé).
_RE_NEW_OBLIG = re.compile(r"\b(must|shall|required to|is required to|obligated to|may not)\b", re.IGNORECASE)
_RE_ADVICE = re.compile(r"\b(you should|we recommend|consult (a|your) (lawyer|attorney)|seek legal advice)\b", re.IGNORECASE)
_RE_NEW_PARTIES = re.compile(r"\b(third party|licensor|licensee|affiliate|subsidiary)\b", re.IGNORECASE)
_RE_DEADLINES = re.compile(r"\b(within \d+|no later than|deadline|days? after)\b", re.IGNORECASE)

# Ces indicateurs ne sont pas des "ground truth" mais aident à structurer l'analyse qualitative.
def risk_flags(expl: str) -> Dict[str, bool]:
    e = expl or ""
    return {
        "mentions_new_obligations": bool(_RE_NEW_OBLIG.search(e)),
        "gives_legal_advice": bool(_RE_ADVICE.search(e)),
        "mentions_parties_terms": bool(_RE_NEW_PARTIES.search(e)),
        "mentions_deadlines": bool(_RE_DEADLINES.search(e)),
        "empty_or_too_short": len(e.split()) < 8,
        "too_long": len(e.split()) > 85,
    }


def normalize_ws(text: str) -> str:
    return " ".join(text.split())

def synthetic_target_from_label(label_name: str) -> str:
    label_name = label_name.replace("_", " ").strip().lower()
    return (
        f"This clause is about {label_name}. "
        "It explains the general meaning of this part of the contract in simpler words. "
        "It does not add requirements beyond what is written in the clause."
    )

def render_chat_prompt_for_generation(tok, clause):
    system = "You are a careful educational legal assistant. Explain clauses for non-lawyers with high faithfulness. Never add new obligations, deadlines, penalties, exceptions, or parties. Do not give legal advice. Answer in 2 to 4 short sentences."
    user = f"Explain this legal clause in plain English:\n{normalize_ws(clause)}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_training_example(tok, clause, label_name):
    target = synthetic_target_from_label(label_name)
    system = "You are a careful educational legal assistant. Explain clauses for non-lawyers with high faithfulness. Never add new obligations, deadlines, penalties, exceptions, or parties. Do not give legal advice. Answer in 2 to 4 short sentences."
    user = f"Explain this legal clause in plain English:\n{normalize_ws(clause)}"
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target}
    ]
    
    full_text = tok.apply_chat_template(messages, tokenize=False) + tok.eos_token
    enc = tok(full_text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    
    prompt_msg = messages[:-1]
    prompt_text = tok.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)
    prompt_enc = tok(prompt_text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    
    input_ids = enc["input_ids"][0]
    labels = input_ids.clone()
    prompt_len = int(prompt_enc["attention_mask"][0].sum().item())
    labels[:prompt_len] = -100
    labels[enc["attention_mask"][0] == 0] = -100
    
    return {"input_ids": input_ids, "attention_mask": enc["attention_mask"][0], "labels": labels}

def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

@torch.no_grad()
def generate_explanation(model, tok, clause: str) -> tuple[str, str]:
    prompt = render_chat_prompt_for_generation(tok, clause)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=96,
        do_sample=LORA_DECODING.do_sample,
        num_beams=LORA_DECODING.num_beams,
        repetition_penalty=LORA_DECODING.repetition_penalty,
        no_repeat_ngram_size=LORA_DECODING.no_repeat_ngram_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen_ids = out[0, enc["input_ids"].shape[1]:]
    return normalize_ws(tok.decode(gen_ids, skip_special_tokens=True).strip()), prompt

# 4. Prétraitement pour l'entraînement : création de prompts et masquage des tokens de la partie "prompt" dans les labels
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    cuda_cleanup()
    print("Device:", DEVICE)

    ds = load_dataset("lex_glue", "ledgar")
    train_ds = ds["train"].shuffle(seed=SEED).select(range(min(TRAIN_MAX_EX, len(ds["train"]))))
    test_ds = ds["test"]
    label_names = ds["train"].features["label"].names

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True).to(DEVICE)

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    train_items = [make_training_example(tok, ex["text"], label_names[int(ex["label"])]) for ex in train_ds]
    loader = DataLoader(train_items, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * math.ceil(len(loader) / GRAD_ACCUM_STEPS)
    sched = get_linear_schedule_with_warmup(optim, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    for epoch in range(EPOCHS):
        optim.zero_grad(set_to_none=True)
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
                optim.zero_grad(set_to_none=True)
                sched.step()

    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    model.eval()
    subset = test_ds.shuffle(seed=SEED).select(range(min(EVAL_N, len(test_ds))))
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for i, ex in enumerate(subset):
            clause = ex["text"]
            expl, prompt = generate_explanation(model, tok, clause)
            row = {
                "i": i,
                "model": MODEL_NAME,
                "adaptation": "LoRA fine-tuning",
                "prompting": "chat prompt (system+user)",
                "decoding": asdict(LORA_DECODING),
                "max_input_tokens": 256,
                "max_new_tokens": 96,
                "clause": clause,
                "prompt": prompt,
                "explanation": expl,
                "risk_flags": risk_flags(expl),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Saved qualitative outputs:", OUTPUT_JSONL)

if __name__ == "__main__":
    main()