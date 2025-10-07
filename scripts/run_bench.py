#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from src.uq.eval import run_eval


def main() -> None:
    result = run_eval(Path("configs/default.yaml"))
    md = "| Metric | Value |\n|---|---|\n"
    for key, value in result.metrics.items():
        md += f"| {key} | {value:.4f} |\n"
    (result.output_dir / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
