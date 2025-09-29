# make_calibration_10k.py
# Build a 10,000-example QA calibration set with standardized fields.

import os
import re
import json
import random
from collections import defaultdict

from datasets import load_dataset, concatenate_datasets
from datasets.features import Sequence
from datasets.features import features as datasets_features

# Older cached dataset metadata still records list features using the legacy
# "List" type, which newer `datasets` releases dropped. Register a fallback so
# we can deserialize those caches without upgrading the global installation.
datasets_features._FEATURE_TYPES.setdefault("List", Sequence)

SEED = 42
random.seed(SEED)

OUT_JSONL = "data/calibration/calibration_10k.jsonl"

# ---------- Normalization / helpers ----------

PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # collapse punctuation/whitespace; keep alnum
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_gsm8k_final_answer(answer_str: str) -> str:
    """
    GSM8K answers often contain rationale + final line like '#### 42'.
    We try to extract the final numeric/string after '####'.
    Fallback: last number in the string; else the tail text.
    """
    if answer_str is None:
        return ""
    # Try explicit #### marker
    m = re.search(r"####\s*(.+)$", answer_str.strip())
    if m:
        return m.group(1).strip()
    # Else last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_str)
    if nums:
        return nums[-1]
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in answer_str.split("\n") if ln.strip()]
    return lines[-1] if lines else answer_str.strip()

def sample_indices(n_total: int, k: int) -> list:
    idxs = list(range(n_total))
    random.Random(SEED).shuffle(idxs)
    return idxs[:k]

# ---------- Standardized record factory ----------

def make_record(
    dataset: str,
    q_text: str,
    answers: list,
    answer_type: str,
    options: list = None,
    split: str = "",
    language: str = "en",
    meta: dict = None,
    source_id: str | int = None,
) -> dict:
    """
    Standard schema:
      - id: unique str
      - dataset: str
      - question: str
      - options: list[str] (empty if open)
      - answers: list[str] (canonical strings; may include multiple acceptable forms)
      - answers_normalized: list[str] (normalized versions)
      - answer_type: 'open'|'choice'|'boolean'|'numeric'
      - split: str
      - language: str
      - meta: dict (dataset-specific notes)
    """
    options = options or []
    meta = meta or {}
    answers = [a for a in (answers or []) if isinstance(a, str) and a.strip()]
    ans_norm = sorted({normalize_text(a) for a in answers if a is not None})

    rec_id = f"{dataset}:{source_id}" if source_id is not None else f"{dataset}:{random.getrandbits(64)}"
    return {
        "id": rec_id,
        "dataset": dataset,
        "question": q_text if q_text is not None else "",
        "options": options,
        "answers": answers,
        "answers_normalized": ans_norm,
        "answer_type": answer_type,
        "split": split,
        "language": language,
        "meta": meta,
    }

# ---------- Loaders for each dataset (train/eval as noted) ----------

def load_truthfulqa():
    # Generation setup; small (~817); use validation split.
    ds = load_dataset("truthful_qa", "generation", split="validation")
    # Fields: question, best_answer, correct_answers(list), incorrect_answers(list)
    def convert(ex, i):
        answers = []
        if ex.get("best_answer"):
            answers.append(ex["best_answer"])
        if ex.get("correct_answers"):
            answers.extend(ex["correct_answers"])
        return make_record(
            dataset="truthfulqa",
            q_text=ex.get("question", ""),
            answers=answers,
            answer_type="open",
            options=[],
            split="validation",
            language="en",
            meta={},
            source_id=i,
        )
    return [convert(ds[i], i) for i in range(len(ds))]

def load_triviaqa():
    # Use rc.nocontext → model must answer from memory
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    # Fields: question, answer:{value, aliases}
    def convert(ex, i):
        ans = []
        if ex.get("answer"):
            if ex["answer"].get("value"):
                ans.append(ex["answer"]["value"])
            if ex["answer"].get("aliases"):
                ans.extend(ex["answer"]["aliases"])
        # Deduplicate
        seen = {}
        answers = []
        for a in ans:
            key = normalize_text(a)
            if key and key not in seen:
                seen[key] = True
                answers.append(a)
        return make_record(
            dataset="triviaqa",
            q_text=ex.get("question", ""),
            answers=answers if answers else [ex["answer"]["value"]] if ex.get("answer", {}).get("value") else [],
            answer_type="open",
            options=[],
            split="train",
            language="en",
            meta={},
            source_id=i,
        )
    return [convert(ds[i], i) for i in range(len(ds))]

def load_commonsenseqa():
    ds = load_dataset("commonsense_qa", split="train")
    # Fields: question, choices:{label(list of 'A'..), text(list)}, answerKey ('A'..)
    def convert(ex, i):
        opts = list(ex["choices"]["text"])
        labels = list(ex["choices"]["label"])  # e.g. ['A','B','C','D','E']
        answer_key = ex.get("answerKey")
        ans_text = []
        if answer_key in labels:
            idx = labels.index(answer_key)
            ans_text = [opts[idx]]
        return make_record(
            dataset="commonsenseqa",
            q_text=ex.get("question", ""),
            answers=ans_text,
            answer_type="choice",
            options=opts,
            split="train",
            language="en",
            meta={"answerKey": answer_key, "labels": labels},
            source_id=i,
        )
    return [convert(ds[i], i) for i in range(len(ds))]

def load_openbookqa():
    ds = load_dataset("openbookqa", "main", split="train")
    # Fields: question_stem, choices:{label,text}, answerKey
    def convert(ex, i):
        opts = list(ex["choices"]["text"])
        labels = list(ex["choices"]["label"])
        answer_key = ex.get("answerKey")
        ans_text = []
        if answer_key in labels:
            idx = labels.index(answer_key)
            ans_text = [opts[idx]]
        return make_record(
            dataset="openbookqa",
            q_text=ex.get("question_stem", ""),
            answers=ans_text,
            answer_type="choice",
            options=opts,
            split="train",
            language="en",
            meta={"answerKey": answer_key, "labels": labels},
            source_id=i,
        )
    return [convert(ds[i], i) for i in range(len(ds))]

def load_ai2_arc():
    # Combine Easy + Challenge train splits into one pool
    ds_e = load_dataset("ai2_arc", "ARC-Easy", split="train")
    ds_c = load_dataset("ai2_arc", "ARC-Challenge", split="train")
    ds = concatenate_datasets([ds_e, ds_c])
    # Fields: question, choices:{label,text}, answerKey
    def convert(ex, i):
        opts = list(ex["choices"]["text"])
        labels = list(ex["choices"]["label"])
        answer_key = ex.get("answerKey")
        ans_text = []
        if answer_key in labels:
            idx = labels.index(answer_key)
            ans_text = [opts[idx]]
        return make_record(
            dataset="ai2_arc",
            q_text=ex.get("question", ""),
            answers=ans_text,
            answer_type="choice",
            options=opts,
            split="train",
            language="en",
            meta={"answerKey": answer_key, "labels": labels, "subset": ex.get("id", "")},
            source_id=i,
        )
    return [convert(ds[i], i) for i in range(len(ds))]

def load_strategyqa():
    ds = load_dataset("tasksource/strategy-qa", split="train")
    # Fields: question, answer (bool)
    def convert(ex, i):
        ans_bool = ex.get("answer", False)
        ans_text = ["yes"] if ans_bool else ["no"]
        return make_record(
            dataset="strategyqa",
            q_text=ex.get("question", ""),
            answers=ans_text,
            answer_type="boolean",  # treated as choice over ["yes","no"]
            options=["yes", "no"],
            split="train",
            language="en",
            meta={},
            source_id=i,
        )
    return [convert(ds[i], i) for i in range(len(ds))]

def load_gsm8k():
    ds = load_dataset("gsm8k", "main", split="train")
    # Fields: question, answer (rationale + final)
    def convert(ex, i):
        final = extract_gsm8k_final_answer(ex.get("answer", ""))
        return make_record(
            dataset="gsm8k",
            q_text=ex.get("question", ""),
            answers=[final] if final else [],
            answer_type="numeric",  # often numeric; we still store as string
            options=[],
            split="train",
            language="en",
            meta={},
            source_id=i,
        )
    return [convert(ds[i], i) for i in range(len(ds))]

# ---------- Assemble pools ----------

def build_pools():
    print("Loading datasets...")
    pools = {
        "truthfulqa": load_truthfulqa(),
        "triviaqa": load_triviaqa(),
        "commonsenseqa": load_commonsenseqa(),
        "openbookqa": load_openbookqa(),
        "ai2_arc": load_ai2_arc(),
        "strategyqa": load_strategyqa(),
        "gsm8k": load_gsm8k(),
    }
    sizes = {k: len(v) for k, v in pools.items()}
    print("Sizes:", sizes)
    return pools, sizes

# ---------- Sampling logic ----------

def plan_allocation(sizes: dict, total_target: int = 10_000, seed: int = SEED) -> dict:
    """
    Equal parts across datasets; if a dataset is too small for its equal share,
    take exactly 50% of that dataset. Then top up from large datasets in
    round-robin until we reach total_target.
    """
    keys = list(sizes.keys())
    n = len(keys)
    base = total_target // n
    remainder = total_target - base * n

    # initial desired counts: distribute remainder to first datasets
    desired = {k: base + (1 if i < remainder else 0) for i, k in enumerate(keys)}

    # apply "50% if too small" rule
    alloc = {}
    for k in keys:
        size = sizes[k]
        if size >= desired[k]:
            alloc[k] = desired[k]
        else:
            alloc[k] = max(1, size // 2)  # exactly 50% (floor), at least 1

    # top-up to reach total_target
    current_total = sum(alloc.values())
    if current_total < total_target:
        # datasets with spare capacity
        spare = {k: sizes[k] - alloc[k] for k in keys}
        order = sorted(keys)  # deterministic
        i = 0
        while current_total < total_target:
            k = order[i % len(order)]
            if spare[k] > 0:
                alloc[k] += 1
                spare[k] -= 1
                current_total += 1
            i += 1
            # guard if truly impossible (shouldn't happen)
            if all(sp <= 0 for sp in spare.values()):
                break
    elif current_total > total_target:
        # trim deterministically
        order = sorted(keys, reverse=True)
        i = 0
        while current_total > total_target:
            k = order[i % len(order)]
            if alloc[k] > 0:
                alloc[k] -= 1
                current_total -= 1
            i += 1

    return alloc

def sample_pool(records: list, k: int) -> list:
    if k >= len(records):
        return records[:]
    idxs = sample_indices(len(records), k)
    return [records[i] for i in idxs]

# ---------- Build and save ----------

def main():
    pools, sizes = build_pools()
    alloc = plan_allocation(sizes, total_target=10_000, seed=SEED)
    print("Allocation:", alloc, "Total:", sum(alloc.values()))

    # Sample from each pool
    sampled = []
    for name, k in alloc.items():
        print(f"Sampling {k} from {name} (pool size {sizes[name]})")
        sampled.extend(sample_pool(pools[name], k))

    # Shuffle final set for balance
    random.Random(SEED).shuffle(sampled)

    # Save JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled)} examples to {OUT_JSONL}")

if __name__ == "__main__":
    main()
