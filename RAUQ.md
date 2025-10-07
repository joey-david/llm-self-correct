
## What RAUQ is (in one paragraph)

**RAUQ** is an unsupervised, single-pass uncertainty score for LLM generation that converts intrinsic self-attention and token probabilities into a robust sequence-level uncertainty. It relies on two insights: (i) a few **uncertainty-aware heads** show a **drop in attention to the immediately-preceding token** at hallucinated positions; (ii) uncertainty should **recurrently propagate** across tokens. With a simple per-layer head-selection, a recurrent confidence update, and length-normalized aggregation, RAUQ matches or beats prior UQ methods while adding **<1% latency**. 

## Method (exact equations we implement)

* For each layer `l`, select head `h*_l` with **maximum mean attention to the previous token** over generated tokens only:
  `h*_l = argmax_h ( 1/(N-1) * Σ_{i=2..N} a_{l,h}(i, i-1) )`. (Use only answer tokens, exclude prompt.) 
* Recurrent per-token confidence (per layer):
  `c_l(1) = P(y_1 | x)`;
  `c_l(i) = α·P(y_i | y_<i, x) + (1-α)·a_{l,h*_l}(i, i-1)·c_l(i-1)` for `i>1`.
  This prevents overconfidence cascades and captures conditional dependence. 
* Layer sequence score (answer tokens only):
  `u_l = -(1/N) * Σ_i log c_l(i)` (length-normalized). 
* Final score: `u = max_{l ∈ L_mid} u_l` (we use the middle third of layers; configurable). Median vs max trade-off is small; **max** captures peak uncertainty spikes and is adopted in the paper. 

### Why “previous token” attention?

Empirically, the **drop** in attention from token `i` to token `i-1` strongly discriminates incorrect from correct generations, far more than deeper lookbacks (`i-2`, `i-3`, …). So we focus on `i-1`. 

### α (alpha) defaults

* Summarization: α≈0 works best; RAUQ leans on attention there.
* QA/MT: α in [0.2, 0.5] works well; we default to **α=0.2** globally and allow per-task overrides in config. 

## Baselines we compare to (fast set)

We ship MSP/Perplexity (single pass) and an **improved Attention Score** that (a) ignores prompt attentions and (b) optionally uses selected heads; both choices materially improve that baseline. We include optional **Semantic Entropy** (sampling) with a clear runtime warning. 

## Metrics

* **PRR** (Prediction Rejection Ratio): area-normalized improvement over random when abstaining on the most-uncertain half of the curve—robust to continuous quality metrics (AlignScore/COMET), not just accuracy. We also report ROC-AUC. 

## Computational profile

RAUQ is **single-pass** (reads attentions & logits already computed) and adds **<1% overhead** vs plain inference in the paper’s H100 setup; sampling-based methods blow up runtime by 4–8×. Our code mirrors that design. 

## Token-local spikes for control

Besides `u`, we expose `s_i = max_l [-log c_l(i)]` so downstream controllers can detect **uncertainty spikes** and intervene mid-generation (rollback/CoT).

## Implementation details that matter

* Use **only generated tokens** when aggregating attentions and confidences (exclude prompt); this change alone improves attention-based baselines and aligns with the paper’s ablations. 
* Choose heads by **mean attention to `i-1`**; averaging over all heads washes out the signal. 
* Aggregate across layers with **max** (configurable to median). 

## Gated interventions (our add-on)

We add a **utility-optimal controller**:

* **Spike detection** on `s_i` (absolute/relative thresholds, rolling statistics).
* **Rollback**: pop `d` tokens and **regenerate** with safer decoding (lower temp/top-p, higher repetition penalty).
* **CoT trigger**: splice a terse CoT prefix and continue.
* **Utility calibration**: choose `{τ_abs, τ_rel, d}` to maximize `U = ΔAcc·V − C_cot − C_rb·d − C_latency·Δt` on a calibration split.

## Repro switches (configs)

* Model, layers subset, α, datasets/subset sizes, decoding, `num_samples` (for Semantic Entropy), gating thresholds, rollback depth, CoT template.

## References

Core method, head-selection, recurrence, aggregation, α behavior, PRR definition and runtime figures are drawn from “Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty Quantification for LLMs (RAUQ)” (preprint). 
