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



# 1. Configuration du Décodage
# On définit deux modes : SAFE (déterministe, moins de risques) 
# et RISKY (échantillonnage, plus fluide mais imprévisible).
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42 
N_EVAL = 30

MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 96

OUTPUT_JSONL = "gpt2_fewshot_exo3_outputs.jsonl"
SUMMARY_JSON = "gpt2_fewshot_exo3_summary.json"


@dataclass
class DecodingConfig:
    name: str
    do_sample: bool
    temperature: float
    top_p: float
    num_beams: int
    repetition_penalty: float
    no_repeat_ngram_size: int


DECODING_MODES: List[DecodingConfig] = [
    DecodingConfig(
        name="safe",
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=2,
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

# 2. Fonctions Utiles : Seed, Nettoyage GPU, Prompting, Masquage, Décodage
def set_seed(seed: int) -> None:
    """Fixe les graines aléatoires pour torch et python afin de garantir la reproductibilité."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_subset(ds, n: int, seed: int):
    """Échantillonne de manière déterministe un sous-ensemble pour évaluation qualitative."""
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    return [ds[i] for i in idxs[: min(n, len(ds))]]


def normalize_ws(text: str) -> str:
    """Normalise l'espace blanc en compressant les multiples espaces en un seul.

    Utile pour nettoyer les clauses ou l'entrée utilisateur avant tokenisation.
    """
    return " ".join(text.split())



def build_fewshot_demos(ds_train) -> List[Dict[str, str]]:
    """Construit deux démonstrations few‑shot à partir des deux premiers exemples d'entraînement.

    Les explications sont rédigées manuellement et sont destinées à montrer
    le style souhaité sans fournir d'informations juridiques réelles.
    """
    ex0 = ds_train[0]["text"]
    ex1 = ds_train[1]["text"]

    demo0_expl = (
        "This clause describes what information must be kept private and who is expected to keep it confidential. "
        "It also explains that the information should not be shared or used beyond what the agreement allows."
    )
    demo1_expl = (
        "This clause explains when the agreement can end and what happens when it ends. "
        "It outlines the general conditions for ending the relationship without adding extra details not in the text."
    )

    return [
        {"clause": normalize_ws(ex0), "explanation": demo0_expl},
        {"clause": normalize_ws(ex1), "explanation": demo1_expl},
    ]

def build_prompt_fewshot(clause: str, demos: List[Dict[str, str]]) -> str:
    clause = normalize_ws(clause)

    header = (
        "You are a careful educational legal assistant.\n"
        "Task: Explain the clause in plain English for a non-lawyer.\n"
        "Rules:\n"
        "- Be faithful to the clause.\n"
        "- Do NOT add new obligations, exceptions, deadlines, penalties, or parties.\n"
        "- Do NOT give legal advice.\n"
        "- If the clause is unclear, say so briefly.\n"
        "- Output 2 to 4 short sentences.\n\n"
        "Here are examples:\n\n"
    )

    demo_txt = ""
    for k, d in enumerate(demos, start=1):
        demo_txt += f"Example {k}\n"
        demo_txt += f"Clause:\n{d['clause']}\n"
        demo_txt += f"Explanation:\n{d['explanation']}\n\n"

    query = (
        "Now explain the next clause.\n"
        f"Clause:\n{clause}\n\n"
        "Explanation:\n"
    )
    return header + demo_txt + query


def decode_generated_only(tok: AutoTokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    """Extrait et décode seulement la partie générée, sans répéter le prompt."""
    gen_ids = output_ids[0, input_ids.shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return normalize_ws(text.strip())


# expressions régulières servant à détecter différents types de risques dans l'explication
_RE_NEW_OBLIG = re.compile(r"\b(must|shall|required to|is required to|obligated to|may not)\b", re.IGNORECASE)
_RE_ADVICE = re.compile(r"\b(you should|we recommend|consult (a|your) (lawyer|attorney)|seek legal advice)\b", re.IGNORECASE)
_RE_NEW_PARTIES = re.compile(r"\b(third party|licensor|licensee|affiliate|subsidiary)\b", re.IGNORECASE)
_RE_DEADLINES = re.compile(r"\b(within \d+|no later than|deadline|days? after)\b", re.IGNORECASE)


def risk_flags(expl: str) -> Dict[str, bool]:
    """Renvoie des drapeaux booléens indiquant la présence de risques heuristiques.

    Les règles simples repèrent obligations, conseils juridiques, mentions de parties,
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

# 3. Génération d'une Explication
@torch.no_grad()
def generate_one(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    decoding: DecodingConfig,
) -> str:
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

    return decode_generated_only(tok, enc["input_ids"], out)

# 4. Boucle Principale : Inférence et Évaluation
def main() -> None:
    set_seed(SEED)
    print("Device:", DEVICE)

    ds = load_dataset("lex_glue", "ledgar")
    train = ds["train"]
    test = ds["test"]
    subset = pick_subset(test, N_EVAL, SEED)

    demos = build_fewshot_demos(train)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    agg = {m.name: {"n": 0, "flags_count": {}} for m in DECODING_MODES}

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for i, ex in enumerate(subset):
            clause = ex["text"]
            prompt = build_prompt_fewshot(clause, demos)

            for decoding in DECODING_MODES:
                explanation = generate_one(model, tok, prompt, decoding)
                flags = risk_flags(explanation)

                agg[decoding.name]["n"] += 1
                for k, v in flags.items():
                    agg[decoding.name]["flags_count"][k] = agg[decoding.name]["flags_count"].get(k, 0) + int(v)

                row = {
                    "i": i,
                    "model": MODEL_NAME,
                    "adaptation": "prompt-based",
                    "prompting": "few-shot (2 shots)",
                    "decoding": asdict(decoding),
                    "max_input_tokens": MAX_INPUT_TOKENS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "demos": demos, 
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