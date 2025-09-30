## RAUQ calculations and generation

This CLI drives a causal LM token-by-token, logging log-probs/prev-token attention so we can score RAUQ uncertainty on the decoded answer.
To run it, either use a YAML config describing the dataset and model settings or use the default one. Example:

```yaml
in: data/calibration/mmlu_calib.jsonl
out: data/runs/calibration_rauq.jsonl
model: Qwen/Qwen3-8B
alpha: 0.2
max_new: 128
device: auto
store_all_heads: false
seed: 42
dataset_fraction: 0.5
```

Run with `python src/rauq_minimal.py --config path/to/config.yaml`. Omitting `--config` falls back to the defaults baked into the script. The runner streams one JSONL line per selected record (respecting `dataset_fraction`) with the decoded text, RAUQ summaries, and optionally all per-head attentions.

### Evaluating RAUQ with PRR

After running `src/calibration/rauq_calib.py`, compute the Prediction Rejection Ratio (PRR) – the
key metric used in the RAUQ paper – via

```
python -m eval.compute_prr --infile data/artifacts/rauq_output.jsonl
```

This reports PRR per dataset using AlignScore (if available) or accuracy, matching the evaluation
protocol described in *Uncertainty-Aware Attention Heads*.

### Focusing on MMLU (paper's strongest QA benchmark)

1. Build a calibration set balanced across MMLU subjects:

   ```
   python src/calibration/compile_calib.py --datasets mmlu --out data/calibration/mmlu_calibration_ds.jsonl
   ```

2. Run RAUQ with the provided config (Qwen3-8B by default):

   ```
   python src/calibration/rauq_calib.py --config configs/mmlu_rauq.yaml
   ```

3. Score the run with PRR:

   ```
   python -m eval.compute_prr --infile data/artifacts/mmlu_rauq_output.jsonl
   ```
