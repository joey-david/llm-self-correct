## RAUQ calculations and generation

This CLI drives a causal LM token-by-token, logging log-probs/prev-token attention so we can score RAUQ uncertainty on the decoded answer.
To run it, either use a YAML config describing the dataset and model settings or use the default one. Example:

```yaml
in: data/calibration/calibration_10k.jsonl
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
