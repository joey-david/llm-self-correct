# TODO

## Finding hallucination threshold

### Record token RAUQs on calibration dataset
- Build calibration dataset as combination of QA, reasoning, etc, datasets ,some with known hallucinations, target 5k - 10k examples.
- For each example, run greedy decoding once with no CoT, and record token RAUQs, assign final label based on example ground truth (possibly using AlignScore).
- For wrong answers, find the token with lowest RAUQ, and label that token as a hallucination trigger.

### Find threshold
- Use logistic regression to map $u_t$ to $P(fail | u_t)$, keeping it per-task (one mapping per type of task: QA, summarization). Let's start only with QA and expand later.


### Choose threshoulds
- Define a utility function like $Util(\theta) = \Delta F1(\theta) - \lambda \cdot \text{ExtraTokens}(\theta) $ (i.e. F1 score gain minus extra tokens generated due to CoT, to maximize efficiency of CoT usage).
- Choose $\theta^* = \arg\max_\theta Util(\theta)$. But keep a trigger rate cap (e.g. 5 triggers/gen%) to avoid too many CoT generations.

## Implement in generation
