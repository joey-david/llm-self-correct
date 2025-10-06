CODEX, THIS IS MEANT FOR YOU: DON'T EVER RUN THE ACTUAL INFERENCE SCRIPTS YOURSELF, THERE'S NO GPU ON THIS MACHINE. You can run whatever else you desire, though.

# PROJECT TODO — Reproducible UQ & CoT Gating (RAUQ or Not)

**Assumptions**

* Python ≥ 3.11; PyTorch ≥ 2.2; transformers ≥ 4.44; datasets; scikit-learn; matplotlib; seaborn disabled; sentence-transformers.
* HF token available via `HF_TOKEN` in `.env`.
* Use **base** Llama-3.1-8B (`meta-llama/Meta-Llama-3.1-8B`), greedy decoding, **no chat template**.
* Deterministic seed = `20251006` used everywhere.
* All paths relative to repo root.

---

## 0) Environment & Repo Hygiene

**Tasks**

1. Create `.env` with `HF_TOKEN=...`. DONE
2. Create virtualenv and pin deps.
3. Add a **global run config** YAML.

**Implementation**

* Add `env/requirements.txt` with exact versions (transformers, torch, scikit-learn, sentence-transformers, huggingface_hub, python-dotenv, matplotlib).
* Add `configs/global.yaml`:

  ```yaml
  model: meta-llama/Meta-Llama-3.1-8B
  device: auto
  use_chat_template: false
  do_sample: false
  temperature: 0.0
  top_p: 1.0
  num_beams: 1
  max_new_tokens: 48
  seed: 20251006
  ```
* Add `Makefile` targets:

  * `make venv` → create/activate venv & install requirements
  * `make test` → run unit tests
  * `make run_bench` → run full benchmark suite

**Done when**

* `python -c "import torch, transformers"` works.
* `python -m scripts.smoketest` runs and prints model id + seed.

---

## 1) Data IO & Logging (Deterministic)

**Create**

* `src/uq/io.py`
* `data/experiments/` (artifacts root)

**Implement**

```python
# src/uq/io.py
SCHEMA = {
  "prompt": str, "answer": str, "label_em": int, "label_f1": float, "label_sem": int,
  "mean_nll": float, "min_prob": float, "max_entropy": float, "logit_margin": float,
  "length": int,
  "rauq_u_final": float, "rauq_u_token_max": float,
  "d_prev_attn_max": float, "d_logprob_max": float,
  "layer_heads_selected": dict,  # {layer_index:[head_ids]}
  "model_id": str, "seed": int, "config": dict
}
def write_jsonl(path: str, rows: list[dict]) -> None: ...
def read_jsonl(path: str) -> list[dict]: ...
```

**Done when**

* Writing & reading round-trips 100 rows and validates keys.

---

## 2) Prompting & Generation Adapter

**Create**

* `src/uq/generate.py`

**Implement**

```python
class GenAdapter:
    def __init__(self, cfg: dict): ...
    def encode(self, text: str) -> torch.LongTensor: ...
    def step(self, input_ids, past_key_values=None, attention_mask=None,
             output_attentions: bool=True) -> tuple[int, torch.Tensor | None, tuple[torch.Tensor,...] | None, tuple]:
        """Return (next_token_id, logits, attentions, past). Attentions shape per layer: (1, H, Q, K)."""
    def generate_once(self, prompt: str) -> dict:
        """
        Deterministic greedy decode. Returns:
        {
          "tokens": list[int], "texts": list[str],
          "logprobs": list[float],  # log p(y_i)
          "attn_prev": list[list[float]],  # per-token: per-head a_{i,i-1} at each layer H
          "layer_heads_selected": dict     # filled later
        }
        """
```

**Constraints**

* Use `cfg['do_sample']=False`, no chat template.
* Seed all RNGs with `cfg['seed']` each call.

**Done when**

* For probe prompt `"What is King Henry holding in the Portrait of Henry VIII?"` the adapter returns tokens/logprobs/attentions without error.
* Attentions tensor present for each generated token.

---

## 3) Labels & Baselines

**Create**

* `src/uq/metrics.py`
* `src/uq/baselines.py`

**Implement**

```python
# metrics
def em_label(pred: str, gold: str) -> int: ...
def f1_label(pred: str, gold: str) -> float: ...
def sem_label(pred: str, gold: str, model_name="sentence-transformers/all-MiniLM-L6-v2", tau=0.75) -> int: ...

# baselines
def mean_nll(logprobs: list[float]) -> float: ...
def min_prob(logprobs: list[float]) -> float: ...
def max_entropy(logits: torch.Tensor) -> float: ...  # if logits available; else skip
def logit_margin(top1: float, top2: float) -> float: ...
def answer_length(tokens: list[int]) -> int: ...
```

**Done when**

* Unit tests for exact-match/token F1.
* Baseline features computed for 10 synthetic examples.

---

## 4) RAUQ Extraction — **Fixes + Features**

**Create/Modify**

* `src/uq/rauq.py`

**Directives**

1. **Align attentions to correct token**
   For token `i`, store attentions from the same step that produced `y_i`. Do **not** shift or fill later.
2. **Include token 1** in recurrence and layer scoring.
3. **Head-set selection per layer**
   For each layer `l`, compute mean `a_{i,i-1}` across tokens. Select top-`m=3` heads; store indices. During scoring, average only across these heads.
4. **Temporal features**
   Compute per-token:

   * `Δprev_attn_i = a_{i,i-1} - a_{i-1,i-2}` (0 for i=1)
   * `Δlogprob_i  = p_i - p_{i-1}` (0 for i=1)
     And their 3-token window min/max over i.

**Implement**

```python
@dataclass
class RauqResult:
    u_token: list[float]
    u_final: float
    head_map: dict[int, list[int]]  # layer->heads

def select_heads(attn_prev_per_token: list[np.ndarray], top_m: int=3) -> dict[int, list[int]]: ...
def compute_rauq(attn_prev_per_token: list[np.ndarray], token_probs: list[float],
                 alpha: float=0.2, head_map: dict[int, list[int]]|None=None) -> RauqResult: ...
def temporal_features(attn_prev_per_token: list[np.ndarray], token_probs: list[float]) -> dict[str, float]: ...
```

**Done when**

* Unit test on synthetic attention with an engineered drop at token `k` shows `u_token[k]` peak and `u_final` increases with drop magnitude.
* `head_map` contains ≤ `m` heads per layer, indices valid.

---

## 5) Learned Layer Aggregation (Ablations)

**Create**

* `src/uq/aggregation.py`

**Implement**

```python
def aggregate_max(u_layers: np.ndarray) -> float: ...
def aggregate_topk_mean(u_layers: np.ndarray, k:int=5) -> float: ...
class LearnedLayerWeights:
    def __init__(self, L:int, temperature:float=1.0): ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...  # small logistic or softmax weights over layers
    def score(self, u_layers: np.ndarray) -> float: ...
```

**Done when**

* All three strategies run on the same stored per-layer vectors and produce scalars.

---

## 6) Meta-UQ (Tiny Logistic + Isotonic)

**Create**

* `src/uq/meta.py`

**Implement**

```python
def build_feature_vector(sample: dict) -> np.ndarray:
    # features: [rauq_u_final, mean_nll, min_prob, entropy, logit_margin, length,
    #            d_prev_attn_max, d_logprob_max]
def fit_logistic(X_train, y_train, C=1.0) -> object: ...
def fit_isotonic(scores_train, y_train) -> object: ...
def predict_proba(model, X) -> np.ndarray: ...
```

**Done when**

* On a held-out split the logistic returns calibrated probabilities in [0,1]; isotonic transforms scores monotonically (used later for thresholds/PPV, not AUC).

---

## 7) Gating Policy for CoT (Decision-Theoretic)

**Create**

* `src/gating/policy.py`

**Implement**

```python
def expected_utility(p_err_direct: float, p_err_cot: float,
                     delta_tokens: int, lambda_cost: float, mean_tokens: int) -> float: ...
def trigger_cot(p_err_direct: float, p_err_cot: float,
                delta_tokens: int, lambda_cost: float, mean_tokens: int) -> bool: ...
```

**Done when**

* Unit tests confirm monotonicity w.r.t. `lambda_cost` and `delta_tokens`.

---

## 8) Datasets & Runners

**Create**

* `src/uq/datasets.py` with loaders for: TruthfulQA, SciQ, CoQA (QA); CNN/DailyMail or SAMSum (Summ); WMT14/19 (MT). Provide minimal samples if full sets not locally available.
* `scripts/run_collect.py` — generate answers + features + RAUQ + temporal features; write JSONL.
* `scripts/run_eval.py` — compute metrics/ROC/PR and save plots.
* `scripts/run_gating.py` — compute Expected Utility vs cost curves.

**CLI Contracts**

```bash
# Collect (per dataset)
python -m scripts.run_collect --config configs/global.yaml --dataset truthfulqa --out data/experiments/tqa.jsonl

# Evaluate ROC/PR
python -m scripts.run_eval --in data/experiments/tqa.jsonl --label sem --out data/experiments/tqa_eval/

# Gating curves
python -m scripts.run_gating --direct data/experiments/tqa.jsonl \
  --cot data/experiments/tqa_cot.jsonl --lambda-grid 0.5,1,2,5 \
  --out data/experiments/tqa_gating/
```

**Artifacts**

* JSONL per dataset (`*_direct.jsonl`, `*_cot.jsonl` if CoT collected).
* Plots: `roc.png`, `pr.png`, `eu_vs_cost.png`.
* Tables: CSV with AUC/PR/AUPRC per feature and meta-UQ.

**Done when**

* JSONL sizes > 100 examples per dataset.
* Plots saved; CSV written.

---

## 9) Probe & Unit Tests

**Create**

* `tests/test_synthetic_rauq.py` — deterministic synthetic attention drop.
* `tests/test_henry_probe.py` — run the “Henry VIII” probe and assert:

  * tokens contain `fal` `con` **if and only if** generation matches paper stimulus (skip assert if not present; still check code paths).
* `tests/test_io.py`, `tests/test_policy.py`.

**Done when**

* `pytest -q` passes all tests locally.

---

## 10) Analysis Logic & Acceptance Criteria

**Create**

* `analysis/criteria.md` with **binary decision rules**:

**Keep RAUQ** if **any** of the following on held-out test:

* `AUC(rauq_u_final) >= AUC(mean_nll) + 0.03` **or**
* `AUC(logistic(features_with_rauq)) >= AUC(logistic(features_without_rauq)) + 0.02` **or**
* Decision-theoretic gating with RAUQ-augmented predictor yields **EU gain ≥ 5%** at token-cost ratio 0.3–0.7.

**Drop RAUQ** if **all** are true:

* `AUC(rauq_u_final) < AUC(mean_nll) + 0.01` **and**
* `AUC(logistic(features_with_rauq)) - AUC(logistic(features_without_rauq)) < 0.01` **and**
* EU gain < 3% across all λ grid points.

**Done when**

* A single markdown report (`reports/final.md`) states **KEEP** or **DROP** with plots + tables.

---

## 11) Plots & Tables (Exact Filenames)

* `data/experiments/<dataset>/roc_<feature>.png`
* `data/experiments/<dataset>/pr_<feature>.png`
* `data/experiments/<dataset>/eu_curve.png`
* `data/experiments/<dataset>/summary.csv`
  Columns: `feature, AUC, AUPRC, PPV@10, PPV@20, EU@lambda{0.5,1,2,5}`

**Done when**

* All files exist and are non-empty; CSV contains ≥ 6 features (rauq, mean_nll, min_prob, entropy, margin, meta_logistic).

---

## 12) Run Order (single command list)

1. `make venv`
2. `python -m scripts.smoketest`
3. **Collect direct answers**:

   ```
   python -m scripts.run_collect --config configs/global.yaml --dataset truthfulqa --out data/experiments/tqa_direct.jsonl
   python -m scripts.run_collect --config configs/global.yaml --dataset sciq       --out data/experiments/sciq_direct.jsonl
   python -m scripts.run_collect --config configs/global.yaml --dataset coqa       --out data/experiments/coqa_direct.jsonl
   ```
4. **Collect CoT answers** (identical prompts + “Let’s think step by step.”):

   ```
   python -m scripts.run_collect --config configs/global.yaml --dataset truthfulqa --cot --out data/experiments/tqa_cot.jsonl
   ```
5. **Evaluate**:

   ```
   python -m scripts.run_eval --in data/experiments/tqa_direct.jsonl  --label sem --out data/experiments/tqa_eval/
   python -m scripts.run_eval --in data/experiments/sciq_direct.jsonl --label sem --out data/experiments/sciq_eval/
   python -m scripts.run_eval --in data/experiments/coqa_direct.jsonl --label sem --out data/experiments/coqa_eval/
   ```
6. **Gating** (TruthfulQA at minimum):

   ```
   python -m scripts.run_gating --direct data/experiments/tqa_direct.jsonl \
     --cot data/experiments/tqa_cot.jsonl --lambda-grid 0.5,1,2,5 \
     --out data/experiments/tqa_gating/
   ```
7. **Synthesize decision**:

   * Run `python -m scripts.make_report` → writes `reports/final.md`.
   * Manually verify acceptance criteria in `analysis/criteria.md`.

---

## 13) Common Pitfalls (explicit checks)

* **Chat template accidentally on** → search for `<|start_header_id|>` tokens; must not appear.
* **Attention misalignment** → unit test must confirm per-token attention corresponds to same-step output.
* **Seed drift** → set all RNG seeds before **every** generation batch.
* **AUC expectations** → isotonic calibration **does not change AUC**; only thresholds/PPV. Do not expect AUC shifts from calibration alone.

---

## 14) Final Deliverables

* `data/experiments/**` (JSONL, plots, CSV)
* `reports/final.md` (KEEP or DROP RAUQ, with plots & metrics)
* `src/**` (modules implemented above)
* `tests/**` passing
* `Makefile`, `configs/global.yaml`, `.env.example`

---

### Exit Conditions (binary)

* If **KEEP**: provide the best RAUQ variant (head-set + temporal + learned aggregation) and its **per-dataset AUC/PR** and **EU** curves; include exact hyperparams.
* If **DROP**: provide the best **black-box meta-UQ** (no attentions) and the **decision-theoretic CoT gating** policy with λ sweep; include cost-accuracy Pareto.
