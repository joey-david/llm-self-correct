**You are Codex. Your job is to build a complete, GPU-first, reproducible codebase that:**

1. reproduces RAUQ end-to-end on open LLMs,
2. benchmarks RAUQ vs classical UQ baselines with PRR/ROC-AUC,
3. adds **utility-optimal rollback** and **CoT triggering on uncertainty spikes**, and
4. is safe to execute only on a remote GPU box.

Follow this spec exactly.

## Hard rules

* **Never run inference on CPU.** If `torch.cuda.is_available()` is false, abort with a loud, single-line message and non-zero exit code.
* Default model: `meta-llama/Meta-Llama-3.1-8B` (base), greedy decoding, **no chat template**.
* Determinism: global seed `20251006`. Set `torch.use_deterministic_algorithms(True)` when feasible; set cudnn flags for determinism; document caveats.
* One forward pass per sample for RAUQ (attention + logits) — minimal overhead.
* Use `transformers >= 4.44`, `torch >= 2.2`, Python `>= 3.11`.
* All secrets via env (e.g., `HF_TOKEN`). No secrets in code.

## Repository layout (create all files)

```
.
├── Makefile
├── pyproject.toml
├── README.md
├── RAUQ.md
├── env/
│   └── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── models/llama3_8b.yaml
│   ├── datasets/
│   │   ├── truthfulqa.yaml
│   │   ├── mmlu.yaml
│   │   ├── xsum.yaml
│   │   ├── samsum.yaml
│   │   ├── cnn_dailymail.yaml
│   │   ├── wmt14_fren.yaml
│   │   └── wmt19_deen.yaml
│   └── gating.yaml
├── src/
│   ├── uq/
│   │   ├── rauq.py
│   │   ├── head_select.py
│   │   ├── baselines.py
│   │   ├── metrics.py         # PRR, ROC-AUC, rejection curves
│   │   ├── eval.py            # evaluation orchestration
│   │   ├── io.py              # deterministic IO + logging
│   │   └── gen.py             # HF generation wrapper (attn+logits)
│   ├── datasets/
│   │   ├── truthfulqa.py
│   │   ├── mmlu.py
│   │   ├── xsum.py
│   │   ├── samsum.py
│   │   ├── cnn_dailymail.py
│   │   ├── wmt14_fren.py
│   │   └── wmt19_deen.py
│   └── aspects/
│       ├── gater.py           # spike detection, CoT trigger, rollback policy
│       ├── rollback.py        # token-level rollback & regenerate
│       ├── cot.py             # CoT re-prompting & decoding tweaks
│       └── utility.py         # utility model & threshold calibration
├── scripts/
│   ├── run_bench.py
│   ├── run_eval.py
│   ├── calibrate_gating.py
│   └── plot_uq_vs_alignscore.py
├── tests/
│   ├── test_rauq.py
│   ├── test_head_select.py
│   ├── test_metrics.py
│   ├── test_baselines.py
│   └── test_aspects.py
└── data/
    ├── cache/
    └── experiments/           # all artifacts live here
```

## Environment & tooling

* `pyproject.toml` with `setuptools` + pinned deps; `env/requirements.txt` mirrors versions. Make sure to use uv for fast dependency setup.
* Add `Makefile` targets:

  * `make test` → run unit tests
  * `make run_bench` → reproduce full suite on small subsets (config-driven)
  * `make qa_sanity` / `make summ_sanity` / `make mt_sanity` → tiny smoke runs
* Logging: `loguru` or std `logging` with ISO timestamps; write JSONL artifacts to `data/experiments/`.

## Generation wrapper (`src/uq/gen.py`)

* Build a thin wrapper over HF `AutoModelForCausalLM` + `AutoTokenizer`:

  * Load with `attn_implementation="eager"`, `output_attentions=True`, `return_dict_in_generate=True`, `use_cache=True`, greedy decoding.
  * Return: generated tokens, per-step logits/probs, attentions (list per layer, head, seq).
  * Ensure attentions & probs are aligned only over **generated tokens**; exclude prompt.

## RAUQ core (`src/uq/rauq.py`)

Implement RAUQ exactly as in the paper:

* **Head selection per layer**: choose head with **max mean attention to the immediately preceding token** over generated tokens only.
* **Recurrent confidence per token & layer**
  For token `y_i` (i>1):
  `c_l(i) = α * P(y_i | y_<i, x) + (1-α) * a_l,h*(i, i-1) * c_l(i-1)`;
  for i=1: `c_l(1) = P(y_1 | x)`.
* **Sequence score per layer**: `u_l = -(1/N) * sum_i log c_l(i)` (generated tokens only).
* **Final score**: `u = max_{l ∈ L_mid} u_l` where `L_mid` is the middle third of layers by index (configurable).
* Expose both **sequence score** and **token-local spikes**: define `s_i = max_l [ -log c_l(i) ]` to use for spike gating.

## Baselines (`src/uq/baselines.py`)

Implement fast, reputable baselines with unified API:

* **MSP / Perplexity** (single-pass).
* **Attention Score (improved)**: only generated-token attentions; support “selected-head” variant; exclude prompt attentions; length-normalized.
* **Semantic Entropy (optional sampling)**: configurable `num_samples` (default 6) with temperature; warn about overhead.
* Add stubs for Focus/CCP/SAR/LUQ/EigenScore with “not implemented” warnings, but keep the interfaces so they can be filled later.

## Metrics (`src/uq/metrics.py`)

* **PRR** on first 50% of rejection curve; **ROC-AUC**; plotting helpers.
* Quality metrics:

  * **Accuracy** (MMLU/GSM8k).
  * **AlignScore** for QA/Summ; load `yzha/AlignScore` with a robust import fallback (accept both `alignscore.AlignScore` and `alignscore.alignscore.AlignScore`); cache model.
  * **COMET** for MT (optional; allow skipping if model not installed).

## Datasets

* Minimal, solid loaders for: TruthfulQA, MMLU, XSum, SamSum, CNN/DM, WMT14 fr-en, WMT19 de-en.
* Each loader yields `(id, prompt, gold_answer, task_kind)` and a **small, reproducible subset** configurable by `configs/datasets/*.yaml`.

## Evaluation harness (`src/uq/eval.py`, `scripts/run_bench.py`)

* Pipeline: generate → UQ scores (RAUQ + baselines) → quality → PRR/ROC-AUC → JSONL + plots.
* Save:

  * `.../predictions.jsonl` (prompt, pred_text, gold, per-token probs, per-layer selected head indices),
  * `.../scores_{method}.jsonl` (u, s_i),
  * `.../metrics.json` (PRR, ROC-AUC),
  * `.../figs/*.png`.

## Gating & interventions (`src/aspects`)

* **Spike detector (`gater.py`)**:

  * Accept per-token `s_i` and rolling z-score; declare spike if `s_i > τ_abs` **or** `s_i - median_{k} s_{i-k:i-1} > τ_rel`.
  * Configurable refractory window to avoid thrashing.
* **Interventions**:

  * **Rollback (`rollback.py`)**: on spike at i*, roll back `d` tokens (config grid), then regenerate with safer decoding (`temperature↓`, `top_p↓`, `repetition_penalty↑`, optional constrained decoding).
  * **CoT trigger (`cot.py`)**: on spike, **prepend a terse CoT prompt** and resume from last safe prefix; optionally swap to a “reasoning-friendly” decoding (e.g., min-p sampling with lower entropy).
* **Utility model (`utility.py`)**:

  * Define expected utility:
    `U = ΔAcc * V - C_cot * 1[cot] - C_rb * d - C_latency * (Δt)`
    Calibrate `{τ_abs, τ_rel, d}` by maximizing `U` on a held-out calibration set (`scripts/calibrate_gating.py`).
  * Provide Pareto frontier dump (accuracy vs compute).

## Remote GPU safety

* At program start, assert CUDA + print device name & memory. If missing, **abort** with:
  `GPU_NOT_FOUND: aborting to respect no-CPU-inference policy.`
* Add `--force-cpu` flag only for unit tests that **never** call generation.

## Tests

* Unit tests for RAUQ equations, head selection correctness (including “generated-tokens only”), PRR monotonicity, spike detector edge cases, utility calibration shape.

## Make targets

* `make test`
* `make run_bench` (reads `configs/default.yaml`; runs small, multi-task bench; writes a markdown summary table)
* `make plots` (align RAUQ u vs AlignScore scatter; save figs)

**Deliverables:** working repo matching the tree above, deterministic small-subset runs on a single GPU, and clean artifacts in `data/experiments/`.
