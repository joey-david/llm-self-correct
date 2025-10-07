# RAUQ Reproducible Benchmark

This repository packages a GPU-first pipeline for reproducing **RAUQ** uncertainty scores on open LLMs, benchmarking them against classical baselines, and probing uncertainty-triggered interventions (rollback + chain-of-thought). Everything runs deterministically on top of `meta-llama/Meta-Llama-3.1-8B` with one forward pass per sample.

## Requirements

- Python >= 3.11
- CUDA-capable GPU (inference aborts if `torch.cuda.is_available()` is false)
- Recommended: [`uv`](https://github.com/astral-sh/uv) for dependency management

Provision a virtual environment and install deps:

```bash
uv venv
source .venv/bin/activate
uv pip install -r env/requirements.txt
```

## Quickstart

1. Edit `configs/default.yaml` to tweak datasets, alpha, or output locations.
2. Run the full small-bench sweep:

   ```bash
   make run_bench
   ```

   Artifacts land under `data/experiments/<timestamp>/`:
   - `predictions.jsonl`: prompts, generations, gold answers
   - `scores_rauq.jsonl`: RAUQ sequence + token-level spikes
   - `scores_baselines.jsonl`: MSP and attention scores
   - `metrics.json`: PRR / ROC-AUC / accuracy summary
   - `summary.md`: Markdown table for quick inspection

3. Plot AlignScore vs. RAUQ (requires optional deps):

   ```bash
   make plots PRED=... SCORES=... OUT=...
   ```

## Sanity runs

- `make qa_sanity` — TruthfulQA + MMLU micro split
- `make summ_sanity` — XSum + SamSum snippets
- `make mt_sanity` — WMT14 fr-en + WMT19 de-en samples

Each command writes artifacts to `data/experiments/<task>_sanity/`.

## Calibration & gating

Use `scripts/calibrate_gating.py` with a grid YAML and stats JSON to pick spike thresholds that maximize expected utility (see `configs/gating.yaml` for defaults). Mid-generation spikes come from `token_spikes` in `scores_rauq.jsonl` and feed into rollback/CoT hooks under `src/aspects/`.

## Tests

Run `make test` to execute the lightweight unit suite: RAUQ math, head selection, PRR monotonicity, spike detection, and utility calibration sanity checks.
