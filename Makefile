.PHONY: test run_bench qa_sanity summ_sanity mt_sanity plots

PYTHON ?= python

install:
	uv pip install -r env/requirements.txt

test:
	$(PYTHON) -m pytest

run_bench:
	$(PYTHON) scripts/run_bench.py

qa_sanity:
	$(PYTHON) scripts/run_eval.py --config configs/qa_sanity.yaml --output-dir data/experiments/qa_sanity

summ_sanity:
	$(PYTHON) scripts/run_eval.py --config configs/summ_sanity.yaml --output-dir data/experiments/summ_sanity

mt_sanity:
	$(PYTHON) scripts/run_eval.py --config configs/mt_sanity.yaml --output-dir data/experiments/mt_sanity

plots:
	$(PYTHON) scripts/plot_uq_vs_alignscore.py --predictions "$(PRED)" --scores "$(SCORES)" --output "$(OUT)"
