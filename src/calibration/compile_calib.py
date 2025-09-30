from __future__ import annotations
import argparse, json, random, re
from typing import Dict, List, Optional
from datasets import load_dataset, concatenate_datasets

# Hyperparams / toggles
CFG = {
    "out": "data/calibration/calibration_ds.jsonl",
    "seed": 42,
    # Enable/disable datasets here.
    "enable": {
        "truthfulqa": True,
        "triviaqa": True,
        "commonsenseqa": True,
        "openbookqa": True,
        "ai2_arc": True,
        "strategyqa": True,
        "mmlu": True,
        "gsm8k": False,
        "popqa": False,
        "hotpotqa_distractor": False,
        "squad_v2_unanswerable": False,
    },
    # Optional caps per-dataset (None = take full enabled split)
    "max_per_dataset": {
        "truthfulqa": None,
        "triviaqa": None,
        "commonsenseqa": None,
        "openbookqa": None,
        "ai2_arc": None,
        "strategyqa": None,
        "mmlu": None,
        "gsm8k": None,
        "popqa": None,
        "hotpotqa_distractor": None,
        "squad_v2_unanswerable": None,
    },
    # If you prefer a global target size, set an int here (None = no global cap)
    "total_target": 5000,
    # When True and total_target is set, sample approximately equally per
    # enabled dataset (instead of letting the largest dataset dominate).
    # Any shortfall for a dataset is redistributed to others with supply.
    "balance_by_dataset": True,
}

random.seed(CFG["seed"])
def make_record(
    dataset: str, q: str, answers: List[str], answer_type: str,
    options: Optional[List[str]] = None, split: str = "", lang: str = "en",
    meta: Optional[dict] = None, source_id: Optional[str|int] = None
) -> dict:
    answers_clean: List[str] = []
    seen: set[str] = set()
    for answer in answers or []:
        if not isinstance(answer, str):
            continue
        stripped = answer.strip()
        if not stripped:
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        answers_clean.append(stripped)
    rec_id = f"{dataset}:{source_id}" if source_id is not None else f"{dataset}:{random.getrandbits(64)}"
    return {
        "id": rec_id,
        "dataset": dataset,
        "question": q or "",
        "options": options or [],
        "answers": answers_clean,
        "answer_type": answer_type,  # 'open'|'choice'|'boolean'|'numeric'
        "split": split,
        "language": lang,
        "meta": meta or {},
    }

def extract_gsm8k_final(ans: str) -> str:
    if not ans: return ""
    m = re.search(r"####\s*(.+)$", ans.strip())
    if m: return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
    if nums: return nums[-1]
    lines = [ln.strip() for ln in ans.split("\n") if ln.strip()]
    return lines[-1] if lines else ans.strip()


# Loaders
def load_truthfulqa() -> List[dict]:
    ds = load_dataset("truthful_qa", "generation", split="validation")
    out = []
    for i in range(len(ds)):
        ex = ds[i]
        answers = []
        if ex.get("best_answer"): answers.append(ex["best_answer"])
        if ex.get("correct_answers"): answers += list(ex["correct_answers"])
        out.append(make_record(
            "truthfulqa", ex.get("question",""), answers, "open",
            split="validation", source_id=i
        ))
    return out

def load_triviaqa() -> List[dict]:
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    out = []
    for i in range(len(ds)):
        ex = ds[i]; ans = []
        if ex.get("answer",{}).get("value"): ans.append(ex["answer"]["value"])
        if ex.get("answer",{}).get("aliases"): ans += list(ex["answer"]["aliases"])
        out.append(make_record("triviaqa", ex.get("question",""), ans, "open", split="train", source_id=i))
    return out

def load_commonsenseqa() -> List[dict]:
    ds = load_dataset("commonsense_qa", split="train")
    out = []
    for i in range(len(ds)):
        ex = ds[i]; opts = list(ex["choices"]["text"]); labels = list(ex["choices"]["label"])
        ak = ex.get("answerKey"); ans = [opts[labels.index(ak)]] if ak in labels else []
        out.append(make_record("commonsenseqa", ex.get("question",""), ans, "choice", options=opts, split="train", meta={"answerKey": ak}, source_id=i))
    return out

def load_openbookqa() -> List[dict]:
    ds = load_dataset("openbookqa", "main", split="train")
    out = []
    for i in range(len(ds)):
        ex = ds[i]; opts = list(ex["choices"]["text"]); labels = list(ex["choices"]["label"])
        ak = ex.get("answerKey"); ans = [opts[labels.index(ak)]] if ak in labels else []
        out.append(make_record("openbookqa", ex.get("question_stem",""), ans, "choice", options=opts, split="train", meta={"answerKey": ak}, source_id=i))
    return out

def load_ai2_arc() -> List[dict]:
    e = load_dataset("ai2_arc", "ARC-Easy", split="train")
    c = load_dataset("ai2_arc", "ARC-Challenge", split="train")
    ds = concatenate_datasets([e, c]); out = []
    for i in range(len(ds)):
        ex = ds[i]; opts = list(ex["choices"]["text"]); labels = list(ex["choices"]["label"])
        ak = ex.get("answerKey"); ans = [opts[labels.index(ak)]] if ak in labels else []
        out.append(make_record("ai2_arc", ex.get("question",""), ans, "choice", options=opts, split="train", meta={"answerKey": ak}, source_id=i))
    return out

def load_strategyqa() -> List[dict]:
    ds = load_dataset("tasksource/strategy-qa", split="train")
    out = []
    for i in range(len(ds)):
        ex = ds[i]; ans = ["yes"] if ex.get("answer", False) else ["no"]
        out.append(make_record("strategyqa", ex.get("question",""), ans, "boolean", options=["yes","no"], split="train", source_id=i))
    return out

def load_mmlu() -> List[dict]:
    """Load the MMLU benchmark across subjects as multiple-choice QA."""
    out: List[dict] = []
    # Preferred configuration: combined dataset with explicit split
    try:
        ds = load_dataset("lukaemon/mmlu", "all", split="dev")
    except Exception:
        # Fallback to hendrycksTest subjects if the combined config is unavailable
        subjects = [
            "abstract_algebra",
            "astronomy",
            "college_biology",
            "college_computer_science",
            "college_physics",
            "global_facts",
            "high_school_chemistry",
            "high_school_mathematics",
            "high_school_physics",
            "machine_learning",
        ]
        parts = []
        for subj in subjects:
            try:
                part = load_dataset("hendrycksTest", subj, split="test")
                parts.append(part)
            except Exception:
                continue
        if not parts:
            return out
        ds = concatenate_datasets(parts)

    for idx in range(len(ds)):
        ex = ds[idx]
        question = ex.get("question", "")
        raw_choices = ex.get("choices") or ex.get("options") or []
        options = [str(c) for c in raw_choices]
        answer = ex.get("answer")
        gold: List[str] = []
        if isinstance(answer, str) and options:
            first_char = answer.strip()[:1].upper()
            if first_char and "A" <= first_char <= "Z":
                pos = ord(first_char) - ord("A")
                if 0 <= pos < len(options):
                    gold.append(options[pos])
        elif isinstance(answer, int) and 0 <= answer < len(options):
            gold.append(options[int(answer)])
        meta = {
            "subject": ex.get("subject"),
        }
        out.append(
            make_record(
                "mmlu",
                question,
                gold,
                "choice",
                options=options,
                split=str(ex.get("split", "")),
                meta=meta,
                source_id=idx,
            )
        )
    return out

def load_gsm8k() -> List[dict]:
    ds = load_dataset("gsm8k", "main", split="train")
    out = []
    for i in range(len(ds)):
        ex = ds[i]; final = extract_gsm8k_final(ex.get("answer",""))
        out.append(make_record("gsm8k", ex.get("question",""), [final] if final else [], "numeric", split="train", source_id=i))
    return out

def load_popqa() -> List[dict]:
    ds = load_dataset("akariasai/PopQA", split="train")
    out = []
    for i in range(len(ds)):
        ex = ds[i]
        q = ex.get("question","")
        # PopQA typically has a single answer string; store as list
        ans = [ex.get("answer","")] if ex.get("answer") else []
        out.append(make_record("popqa", q, ans, "open", split="train", source_id=i))
    return out

def load_hotpotqa_distractor() -> List[dict]:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    out = []
    for i in range(len(ds)):
        ex = ds[i]
        # closed-book style: use question only, store gold answer(s)
        ans = [ex.get("answer","")] if ex.get("answer") else []
        out.append(make_record("hotpotqa_distractor", ex.get("question",""), ans, "open", split="validation", source_id=i))
    return out

def load_squad_v2_unanswerable() -> List[dict]:
    # keep only unanswerable examples from validation for abstention calibration
    ds = load_dataset("squad_v2", split="validation")
    out = []
    for i in range(len(ds)):
        ex = ds[i]
        if ex.get("is_impossible", False):
            out.append(make_record("squad_v2_unanswerable", ex.get("question",""), [], "open", split="validation", meta={"is_impossible": True}, source_id=i))
    return out


# Assembly
LOADERS = {
    "truthfulqa": load_truthfulqa,
    "triviaqa": load_triviaqa,
    "commonsenseqa": load_commonsenseqa,
    "openbookqa": load_openbookqa,
    "ai2_arc": load_ai2_arc,
    "strategyqa": load_strategyqa,
    "mmlu": load_mmlu,
    "gsm8k": load_gsm8k,
    "popqa": load_popqa,
    "hotpotqa_distractor": load_hotpotqa_distractor,
    "squad_v2_unanswerable": load_squad_v2_unanswerable,
}

def cap_sample(lst: List[dict], k: Optional[int], seed: int) -> List[dict]:
    if k is None or k >= len(lst): return lst
    rng = random.Random(seed); idx = list(range(len(lst))); rng.shuffle(idx)
    return [lst[i] for i in idx[:k]]

def main():
    ap = argparse.ArgumentParser(description="Build calibration JSONL with standardized schema.")
    ap.add_argument("--out", default=CFG["out"])
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated dataset keys to include (defaults to CFG['enable']).",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.datasets:
        include = {name.strip() for name in args.datasets.split(",") if name.strip()}
    else:
        include = {name for name, enabled in CFG["enable"].items() if enabled}

    # Load per‑dataset pools first so we can balance sampling later
    pools = {}
    for name, fn in LOADERS.items():
        if name not in include:
            continue
        try:
            recs = fn()
        except Exception as exc:
            print(f"[compile_calib] Skipping dataset '{name}' due to load error: {exc}")
            continue
        recs = cap_sample(recs, CFG["max_per_dataset"].get(name), args.seed)
        if not recs:
            print(f"[compile_calib] No records for dataset '{name}' after capping; skipping.")
            continue
        pools[name] = recs
    # Assemble final list, optionally balancing by dataset
    all_rec = []
    if CFG["total_target"]:
        target = int(CFG["total_target"]) or 0
        names = [n for n in pools.keys()]
        rng.shuffle(names)
        if CFG.get("balance_by_dataset", False) and names:
            # Start with equal quotas, then redistribute leftover capacity
            quotas = {n: min(len(pools[n]), target // len(names)) for n in names}
            assigned = sum(quotas.values())
            # Remainder: round‑robin give to datasets with remaining supply
            remaining = target - assigned
            cycle = [n for n in names]
            while remaining > 0 and cycle:
                nxt = cycle.pop(0)
                if quotas[nxt] < len(pools[nxt]):
                    quotas[nxt] += 1
                    remaining -= 1
                    cycle.append(nxt)
            # Sample according to quotas
            for n in names:
                lst = list(pools[n])
                rng.shuffle(lst)
                all_rec.extend(lst[:quotas.get(n, 0)])
        else:
            # Unbalanced: simply pool everything and take first N after shuffle
            tmp = []
            for n in pools:
                tmp.extend(pools[n])
            rng.shuffle(tmp)
            all_rec = tmp[:target]
    else:
        # No total target: keep everything (still shuffle for downstream variety)
        for n in pools:
            all_rec.extend(pools[n])
        rng.shuffle(all_rec)

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rec:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rec)} examples to {out_path}")

if __name__ == "__main__":
    main()
