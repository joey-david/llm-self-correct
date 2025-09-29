# RAUQ calculations and generation

This CLI drives a causal LM token-by-token, logging log-probs/prev-token attention so we can score RAUQ uncertainty on the decoded answer.

To run it, drop a YAML config describing the dataset and model settings, then call the runner with that config. Example:

```yaml
in: data/calibration/calibration_10k.jsonl
out: data/runs/calibration_rauq.jsonl
model: Qwen/Qwen3-8B
alpha: 0.3
max_new: 48
device: auto
store_all_heads: false
seed: 42
```

Run with `python src/rauq_minimal.py --config path/to/config.yaml`. The script streams one JSONL line per record with the decoded text, RAUQ summaries, and (optionally) all head attentions.
