# Codex Directives

## Mission
Build and maintain the RAUQ uncertainty benchmarking suite exactly as specified in `todo.md`, prioritizing GPU-only execution, reproducibility, and minimal yet clear implementations.

## Hard Rules
- Never run generation on CPU; abort loudly with `GPU_NOT_FOUND: aborting to respect no-CPU-inference policy.`
- Default model: `meta-llama/Meta-Llama-3.1-8B`, greedy decoding, no chat template.
- Always seed with `20251006`, enable deterministic Torch + cuDNN flags, and document caveats.
- Require `torch>=2.2`, `transformers>=4.44`, Python `>=3.11`.
- Keep secrets in environment variables only.

## Engineering Principles
- Favor small, deterministic dataset subsets defined in configs.
- One forward pass per sample for RAUQ and baselines unless sampling is explicitly requested.
- Logging uses ISO timestamps and writes JSONL artifacts under `data/experiments/`.
- Maintain thin abstractions and minimal code while preserving clarity and testability.
- Preserve utility-optimal gating hooks (spike detection, rollback, CoT) and calibrate via scripts.
