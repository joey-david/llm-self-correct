
# RAUQ-Triggered Chain-of-Thought (CoT) Experiments

## Research Idea

This repo should implement and evaluate a simple and powerful idea:

**Enhance LLM reasoning by using RAUQ uncertainty scores as live triggers for rollback + Chain-of-Thought (CoT) deliberation.**

- **RAUQ** (Recurrent Attention-based Uncertainty Quantification) estimates per-token uncertainty directly from the model’s attention heads, with very low compute overhead.  
- **Trigger rule:** When RAUQ exceeds a learned threshold, the decoding loop **rolls back to a safe anchor** and switches the model into CoT mode. It then resamples until a lower-uncertainty continuation emerges.  
- **Stopping rule:** While in CoT, if RAUQ falls below the threshold for a stable window of tokens, the controller exits CoT mode and returns to normal decoding.  

This gives the model multiple “second chances” at its most fragile prediction steps, hopefully improving correctness without brute-force self-consistency or always-on CoT.

Two phases:
1. **v1 (inference-only)**: wrap an existing model with the RAUQ controller — *no training required*.  
2. **v2 (training)**: QLoRA finetune the model on traces generated with this controller, to encourage shorter, more efficient CoTs and reduce inference overhead.

---


## Setup Instructions

1. **Environment**  
   ```bash
   conda create -n rauqcot python=3.10
   conda activate rauqcot
   pip install torch transformers accelerate bitsandbytes vllm datasets evaluate
   ```

2. **Model downloading (live, to save GPU time)**

   * Do not bundle weights into the repo.
   * Use Hugging Face hub to stream models only when needed:

     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer
     model_name = "Qwen/Qwen2.5-7B-Instruct"
     tok = AutoTokenizer.from_pretrained(model_name)
     model = AutoModelForCausalLM.from_pretrained(
         model_name,
         device_map="auto",
         torch_dtype="auto"
     )
     ```

---

## Core Experiment (v1: Inference-only)

* **Controller loop** (`controller.py`):

  1. Decode one token.
  2. Compute RAUQ score for that step.
  3. If score < θ → continue greedy/beam search.
  4. If score ≥ θ → rollback K tokens, enter CoT mode:

     * Sample `m` candidate continuations with a CoT prefix.
     * Score them with RAUQ (and optionally semantic entropy).
     * Commit the candidate with lowest uncertainty.
  5. Exit CoT once RAUQ stays < θ for W consecutive tokens.

* **Training the threshold θ**:

  * Use a held-out calibration set.
  * Fit logistic regression from RAUQ score → error risk.
  * Pick θ that optimizes accuracy–cost trade-off.

* **Datasets / benchmarks**:

  * GSM8K & MATH (math reasoning)
  * HumanEval (code gen)
  * TriviaQA or NaturalQuestions (QA)

* **Metrics**:

  * Accuracy (per dataset)
  * Token cost (tokens per answer)
  * Latency (s/answer)

---

## Ablations

1. **Trigger signal**: RAUQ vs. entropy vs. logit margin.
2. **Threshold**: fixed vs. learned.
3. **Repair strategy**: rollback+CoT vs. pause+rerank (AdaDec-style).
4. **CoT length**: unlimited vs. max 20 tokens vs. RAUQ-based stop.
5. **Rollback depth K**: 1, 3, or dynamic anchor.

Each ablation can be run via `ablations.py` with flags, e.g.:

```bash
python scripts/ablations.py --trigger rauq --repair cot --rollback 3
```

---

## Extension (v2: QLoRA Finetune)

Goal: train the model itself to **internalize uncertainty-triggered CoT**, so inference is cheaper.

* **Data creation**:

  * Use the v1 controller to generate traces.
  * Annotate spans: `[<uncertain> CoT ... Final Answer </uncertain>]`.
  * Keep short, decisive CoTs (truncate rambles).

* **Training** (`finetune_qLoRA.py`):

  * Model: Qwen2.5-7B-Instruct (upgrade to 14B if resources allow).
  * Method: QLoRA (4-bit, rank 16).
  * Optimizer: 8-bit Adam, bf16 training.
  * Context length: 2048.
  * Batch size: tune to GPU, accumulate if needed.
  * Run on A100 40GB with gradient checkpointing.

* **Hypothesis**: Finetuned model learns to switch into “micro-CoTs” at just the right points, reducing the need for heavy external control.

---

## Testing / Remote GPU Tips

* Always **download weights on-the-fly** inside the container (`download_model.py`) to avoid paying idle GPU time while syncing checkpoints.
* For evaluation, pre-download datasets locally with `datasets` library (no GPU needed).
* Save only **LoRA adapters**, not full model weights, to minimize upload/download.