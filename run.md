# RAUQ-CoT Runbook

This is a runbook capturing the commands for the RAUQ-triggered rollback + Chain-of-Thought controller. It covers environment prep, offline calibration, interactive runs, batch evaluation, parameter sweeps, dataset tweaks, and how to interpret the artifacts you get back.

---

## 1. Environment & Repo Prep

1. **Create a Python environment (3.10+) and install deps.**
   ```bash
   conda create -n rauqcot python=3.10
   conda activate rauqcot
   pip install torch transformers accelerate bitsandbytes datasets evaluate scikit-learn
   ```
   *`scikit-learn` is needed by `fit_threshold.py` for logistic calibration.*

2. **Add the repo to `PYTHONPATH` for all CLI calls.**
   ```bash
   export PYTHONPATH=$(pwd)
   mkdir -p artifacts  # default output bucket for calibration assets
   ```

3. **(Optional) Configure Hugging Face cache/token.**
   ```bash
   export HF_HOME=/path/to/hf-cache
   huggingface-cli login  # if the model needs auth
   ```

---

## 2. Data Preparation

### 2.1 Calibration/training splits

All offline steps expect simple `(prompt, reference)` pairs. Two accepted formats:

- **JSONL** line: `{"prompt": "Question?", "completion": "Answer"}` (`reference` or `target` also accepted).
- **TSV** line: `prompt<TAB>reference`.

Keep completions non-empty so the head selector can compute attention drops.

Example helper to dump a GSM8K calibration shard:
```python
from datasets import load_dataset
from pathlib import Path

calib = load_dataset("gsm8k", "main", split="train[:256]")
with Path("data/calibration/gsm8k_calib.jsonl").open("w") as fp:
    for row in calib:
        fp.write(f"{{\"prompt\": \"{row['question']}\\nAnswer:\", \"completion\": \"{row['answer']}\"}}\n")
```

### 2.2 Evaluation datasets

For scripted sweeps you can rely on `ucot.data.benchmarks.load_benchmark`. Supported keys today: `gsm8k`, `math`, `humaneval`. May add more later.

---

## 3. Offline Calibration Workflow

Run these once per model (and per alpha if you plan to vary it a lot).

### 3.1 Select uncertainty-aware heads
```bash
PYTHONPATH=. python scripts/select_heads.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --calibration data/calibration/gsm8k_calib.jsonl \
  --num-examples 512 \
  --output artifacts/qwen25_heads.json \
  --device cuda
```
Outputs `artifacts/qwen25_heads.json` with a dict of `layer -> head` plus the layer subset used at runtime.

### 3.2 Fit the RAUQ trigger threshold θ
```bash
PYTHONPATH=. python scripts/fit_threshold.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --calibration data/calibration/gsm8k_calib.jsonl \
  --heads artifacts/qwen25_heads.json \
  --alpha 0.3 \
  --max-samples 2048 \
  --output artifacts/qwen25_theta.json \
  --device cuda
```
The script runs greedy decoding over the calibration set, collects token-level RAUQ, fits a logistic regressor, and picks θ via the Youden J rule. Keep the JSON for inference runs.

> Re-run this step if you change α, swap datasets/tasks, or observe obvious over/under-triggering.

---

## 4. Interactive / Single-Prompt Runs

### 4.1 Quick sanity check
```bash
PYTHONPATH=. python scripts/run_controller.py \
  --prompt "A man has 3 apples..." \
  --model Qwen/Qwen2.5-7B-Instruct \
  --heads artifacts/qwen25_heads.json \
  --threshold artifacts/qwen25_theta.json \
  --alpha 0.3 \
  --max-new-tokens 256 \
  --device cuda
```
Key overrides:
- `--theta` lets you bypass the stored threshold (e.g., for sweeps).
- `--rollback`, `--cooldown`, `--stability-window` tune rollback behaviour.
- `--cot-length`, `--cot-horizon`, `--cot-temperature`, `--cot-top-p` adjust micro-CoT sampling.

Logs include the completion, number of triggers, and the leading RAUQ trace tail. The script does not persist traces; capture stdout or wrap via Python if you need structured logs.

### 4.2 Inspecting a run programmatically
```python
PYTHONPATH=. python - <<'PY'
from ucot.controller import RAUQController
from ucot.config import ControllerConfig, RAUQConfig
from ucot.rauq import RAUQScorer
from ucot.threshold import ThresholdResult
from ucot.utils.model import load_model

prompt = "If 12x = 3, what is x?"
model_name = "Qwen/Qwen2.5-7B-Instruct"
heads_path = "artifacts/qwen25_heads.json"
thresh_path = "artifacts/qwen25_theta.json"
alpha = 0.3

loaded = load_model(model_name, device="cuda")
scorer = RAUQScorer.from_config(RAUQConfig(alpha=alpha, head_indices_path=heads_path, device="cuda"),
                                num_layers=loaded.model.config.num_hidden_layers)
controller_cfg = ControllerConfig(model_name=model_name, theta=ThresholdResult.load(thresh_path).theta, max_new_tokens=256)
controller_cfg.rollback.rollback_depth = 2
controller_cfg.cot.max_cot_tokens = 20

controller = RAUQController(loaded=loaded, config=controller_cfg, scorer=scorer,
                            threshold=ThresholdResult.load(thresh_path))
output = controller.generate(prompt)
print("Completion:\n", output.completion)
print("Triggers:", len(output.trigger_events))
print("RAUQ (first 10):", output.rauq_scores[:10])
PY
```
This is the pattern you can reuse for notebooks or custom evaluation scripts.

---

## 5. Batch Evaluation & Testing Different Params

### 5.1 Evaluate a benchmark split
```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
from statistics import mean

from ucot.config import ControllerConfig, RAUQConfig
from ucot.controller import RAUQController
from ucot.data.benchmarks import load_benchmark
from ucot.experiments.metrics import METRICS
from ucot.rauq import RAUQScorer
from ucot.threshold import ThresholdResult
from ucot.utils.model import load_model

MODEL = "Qwen/Qwen2.5-7B-Instruct"
ALPHA = 0.3
HEADS = Path("artifacts/qwen25_heads.json")
THETA = Path("artifacts/qwen25_theta.json")

loaded = load_model(MODEL, device="cuda")
rauq = RAUQScorer.from_config(RAUQConfig(alpha=ALPHA, head_indices_path=HEADS, device="cuda"),
                              num_layers=loaded.model.config.num_hidden_layers)
controller_cfg = ControllerConfig(model_name=MODEL, theta=ThresholdResult.load(THETA).theta, max_new_tokens=256)
controller_cfg.rollback.rollback_depth = 2
controller_cfg.cot.max_cot_tokens = 20

controller = RAUQController(loaded=loaded, config=controller_cfg, scorer=rauq,
                            threshold=ThresholdResult.load(THETA))
samples = load_benchmark("gsm8k", limit=50)
metric = METRICS["gsm8k"]
results = []
for sample in samples:
    out = controller.generate(sample.prompt)
    ok = metric({"reference": sample.reference}, out.completion)
    results.append({
        "id": sample.metadata.get("id"),
        "correct": ok,
        "triggers": len(out.trigger_events),
        "extra_tokens": out.extra_tokens,
    })

acc = mean(r["correct"] for r in results)
avg_triggers = mean(r["triggers"] for r in results)
print(f"Accuracy: {acc:.3f}  |  Avg triggers: {avg_triggers:.2f}")
PY
```
Dump `results` to JSON/CSV if you want detailed inspection.

### 5.2 Parameter sweeps

Loop over the knobs you care about (`alpha`, `theta`, `rollback`, CoT length, etc.) and log metrics. Example shell loop for rollback depth:
```bash
for K in 1 2 3; do
  PYTHONPATH=. python scripts/run_controller.py \
    --prompt prompts/gsm8k_example.txt \
    --model Qwen/Qwen2.5-7B-Instruct \
    --heads artifacts/qwen25_heads.json \
    --threshold artifacts/qwen25_theta.json \
    --rollback $K \
    --alpha 0.3 \
    --max-new-tokens 256 | tee logs/run_K${K}.log
done
```
When doing full sweeps, prefer the Python evaluation harness (previous section) so you can summarise metrics programmatically.

---

## 6. Understanding & Using Outputs

- `artifacts/qwen25_heads.json`: layer→head map plus which layers were kept (see `HeadSelectionResult` in `ucot/head_selection.py`).
- `artifacts/qwen25_theta.json`: logistic fit coefficients and the chosen θ (see `ThresholdResult` in `ucot/threshold.py`).
- `ControllerOutput` (returned by Python API):
  - `completion`: decoded text.
  - `trigger_events`: list of `{position, rauq}` objects where rollbacks fired.
  - `rauq_scores`: per-token RAUQ stream (NaNs appear for forced prefix tokens).
  - `extra_tokens`: number of generated tokens beyond the prompt.
  - `stats["triggers"]`: shortcut for trigger count.

Persist `ControllerOutput.__dict__` to JSON if you want post-hoc analysis (trigger histograms, RAUQ trace plots, etc.).

---

## 7. Customising Datasets & Benchmarks

1. **Add a new HF dataset**: edit `ucot/data/benchmarks.py` to register a formatter and dataset metadata. Follow the `gsm8k` example: define a `_format_*` function that returns `BenchmarkSample` and add an entry to `_LOADERS` and `_DATASETS`.
2. **Use custom local files**: point `--calibration` or your evaluation script to your JSONL/TSV files. `load_prompt_completion_pairs` (in `ucot/data/utils.py`) will ingest them without code changes.
3. **Task-specific metrics**: extend `ucot/experiments/metrics.py` if exact match is not enough. Wire the new metric into your evaluation loop.

---

## 8. Ablations & What-If Studies

Use `scripts/ablations.py` to sweep the configurations called out in `technicals.md`:

```bash
PYTHONPATH=. python scripts/ablations.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --benchmark gsm8k \
  --trigger rauq \
  --repair cot \
  --rollback 3 \
  --heads artifacts/qwen25_heads.json \
  --threshold artifacts/qwen25_theta.json \
  --output artifacts/ablations/gsm8k_rauq_cot.json
```

Key flags:
- `--trigger {rauq,entropy,margin}` toggles the uncertainty signal.
- `--threshold-type {learned,fixed}` plus `--theta` lets you pit calibrated vs fixed cutoffs.
- `--repair {cot,rerank,none}` chooses rollback+CoT, pause+rerank (AdaDec-style), or no repair.
- `--cot-policy {rauq,max20,unlimited}` compares RAUQ-based stopping, capped CoT, or effectively unbounded CoT tokens.
- `--rollback` with `--rollback-mode {fixed,anchor}` sweeps rollback depth K, including the dynamic anchor-only option.

The script reports accuracy (`performance`), accuracy per generated token, average token usage, trigger rate, and average latency; raw per-sample stats land in the optional `--output` JSON for downstream plotting.

---

## 9. Towards v2 Finetuning (QLoRA)

The repo ships only the config scaffolding (`FinetuneConfig` in `ucot/config.py`). To prototype v2:

1. **Collect traces**: run the controller with logging that saves `{prompt, cot_span, final_answer, trigger_positions}` whenever a rollback fires.
2. **Curate micro-CoT data**: filter to short, successful repairs and store them under `data/finetune/` (JSONL with fields like `prefix`, `cot`, `answer`).
3. **Train**: adapt an external LoRA script (e.g., `peft` or `trl`). Instantiate `FinetuneConfig` to keep hyperparameters consistent, but you will need to write a driver (e.g., based on `transformers.Trainer`).
4. **Evaluate**: drop the controller back to greedy decoding and check whether triggers fall while accuracy holds.

Until a first-party script lands, treat this as an advanced/optional path.

---

## 10. Troubleshooting

- **`ModuleNotFoundError: ucot`**: ensure `PYTHONPATH` points at the repo root or install editable with `pip install -e .` (if you add a `pyproject.toml`).
- **Empty RAUQ scores**: check that your calibration files contain non-empty completions; head selection skips empty rows.
- **Logistic calibration failure**: confirm `scikit-learn` is installed and you have enough calibration tokens (`--max-samples` > ~1000 recommended).
- **Slow runs**: reduce `--cot-length`, `--cot-horizon`, or `--max-new-tokens`; double-check that you are not using an overly large `num-examples` during head selection.

Happy experimenting!
