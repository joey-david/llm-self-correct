generate a calibration dataset from many qa and other sources.
run the model on it to get rauq values, check answer correctness with heuristics and alignscore.
from rauq and correctness, establish mapping from rauq to p(fail|u_t) per task.
choose threshold to maximize utility function (f1 gain - lambda * extra tokens)
implement in generation code to trigger cot when rauq below threshold.
adjust value of lambda to get optimal trigger rate. Plot F1 gain/token vs lambda, find optimal lambda. Compare vs always cot and never cot baselines, and vs sota.
run on benchmarks to see improvement, don't highlight sheer performance but efficiency of cot usage (f1 gain per extra token).
#### extensions
finetune with qlora for cot use mid-generation and not simply at the start.
add more tasks (summarization, etc)
run multiple cot generations and pick best with alignscore (mcts?)

### Plots / Diagnostics
- After generating `data/artifacts/rauq_output.jsonl`, run:
  `python data/plot_rauq_dashboard.py --infile data/artifacts/rauq_output.jsonl --outdir data/plots`
- This saves a suite of graphs (ROC/PR, histograms, per-dataset AUC/accuracy, token-level dynamics, layer/head distributions, gating trade-offs) and a short `insights.txt` under `data/plots`.
