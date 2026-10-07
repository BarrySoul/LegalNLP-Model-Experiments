#Souleymane Barry
#Maël Fauquette
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Configuration du Modèle et Décodage
# On utilise TinyLlama-1.1B-Chat
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
N_EVAL = 30

MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 96

OUTPUT_JSONL = "tinyllama_exo3_outputs.jsonl"
SUMMARY_JSON = "tinyllama_exo3_summary.json"


@dataclass
class DecodingConfig:
    name: str
    do_sample: bool
    temperature: float
    top_p: float
    num_beams: int
    repetition_penalty: float
    no_repeat_ngram_size: int


# Deux modes de décodage pour comparer : SAFE (déterministe) vs RISKY (échantillonnage)
DECODING_MODES: List[DecodingConfig] = [
    DecodingConfig(
        name="safe",
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        repetition_penalty=1.10,
        no_repeat_ngram_size=3,
    ),
    DecodingConfig(
        name="risky",
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_beams=1,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
    ),
]

# 2. Utilitaires de Prompting et Décodage
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_subset(ds, n: int, seed: int):
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    return [ds[i] for i in idxs[: min(n, len(ds))]]


def normalize_ws(text: str) -> str:
    return " ".join(text.split())

# 3. Prompting spécifique pour TinyLlama
def build_chat_prompt(clause: str) -> str:
    clause = normalize_ws(clause)
    return (
        "<|system|>\n"
        "You are a careful educational legal assistant.\n"
        "You explain clauses for non-lawyers with high faithfulness.\n"
        "You never add new obligations, deadlines, penalties, exceptions, or parties.\n"
        "You do not give legal advice.\n"
        "If unclear, you say so briefly.\n"
        "You answer in 2 to 4 short sentences.\n"
        "</s>\n"
        "<|user|>\n"
        f"Explain this legal clause in plain English:\n{clause}\n"
        "</s>\n"
        "<|assistant|>\n"
        "Explanation:\n"
    )

# 4. Décodage pour extraire uniquement la partie générée (après le prompt)
def decode_generated_only(tok: AutoTokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    gen_ids = output_ids[0, input_ids.shape[1]:]
    return normalize_ws(tok.decode(gen_ids, skip_special_tokens=True).strip())

# Heuristiques de risques (mêmes que dans les autres scripts pour cohérence)
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

# 5. Génération d'une explication à partir d'une clause
@torch.no_grad()
def generate_one(model, tok, clause: str, decoding: DecodingConfig) -> Tuple[str, str]:
    prompt = build_chat_prompt(clause)
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
    ).to(DEVICE)

    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=decoding.do_sample,
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        num_beams=decoding.num_beams,
        early_stopping=True,
        repetition_penalty=decoding.repetition_penalty,
        no_repeat_ngram_size=decoding.no_repeat_ngram_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    explanation = decode_generated_only(tok, enc["input_ids"], out)
    return prompt, explanation

# 6. Main : Chargement du modèle, itération sur les exemples, génération et sauvegarde des résultats
def main() -> None:
    set_seed(SEED)
    print("Device:", DEVICE)

    ds = load_dataset("lex_glue", "ledgar")
    subset = pick_subset(ds["test"], N_EVAL, SEED)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()

    agg = {m.name: {"n": 0, "flags_count": {}} for m in DECODING_MODES}

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for i, ex in enumerate(subset):
            clause = ex["text"]

            for decoding in DECODING_MODES:
                prompt, explanation = generate_one(model, tok, clause, decoding)
                flags = risk_flags(explanation)

                agg[decoding.name]["n"] += 1
                for k, v in flags.items():
                    agg[decoding.name]["flags_count"][k] = agg[decoding.name]["flags_count"].get(k, 0) + int(v)

                row = {
                    "i": i,
                    "model": MODEL_NAME,
                    "adaptation": "prompt-based",
                    "prompting": "zero-shot",
                    "decoding": asdict(decoding),
                    "max_input_tokens": MAX_INPUT_TOKENS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "clause": clause,
                    "prompt": prompt,
                    "explanation": explanation,
                    "risk_flags": flags,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"[{i+1}/{len(subset)}] done")

    summary = {"model": MODEL_NAME, "n_eval": N_EVAL, "seed": SEED, "modes": {}}
    for mode, d in agg.items():
        n = max(1, d["n"])
        summary["modes"][mode] = {
            "n_generations": n,
            "flag_rates": {k: v / n for k, v in d["flags_count"].items()},
        }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved:", OUTPUT_JSONL)
    print("Saved:", SUMMARY_JSON)


if __name__ == "__main__":
    main()