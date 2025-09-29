# Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty Quantification for LLMs

**__Key Idea__: before hallucinations, there's a drop in attention scores to previous tokens in specific, "uncertainty-aware" heads. Track the heads, and define a threshold for the attention scores to previous tokens. If the score drops below the threshold, flag a hallucination.**

## 1. Identify Uncertainty-Aware Heads
For a sentence containing a hallucination, find the heads with the greatest drop in attention scores to previous tokens during the generation of the hallucinated token. <br>
For this, sort answers by AlignScore (semantic proximity to ground truth) and select the top-k and bottom-k as correct and incorrect, respectively.<br>
Then, compute the avg attention weight to the previous token using the attention heads in the later layers (22-29 in llama3.1-8B), i.e. if $N$ is the length of the generated answer, and $a_{i,j}^{lh}$ is the attention weight from token $i$ to token $j \leq i$ in layer $l$ and head $h$, then compute: 
 $$\overline{a}^{lh} = \frac{1}{N-1} \sum_{i=2}^{N}{a_{i,i-1}^{lh}}$$

Then, select the top-m (1 is often best) heads with the highest mean attention weight to the previous token.

## 2. Transmit confidence scores layer-wise.

After having selected 
$
h_l(y) = \underset{h \in \{1, \ldots, H\}}{\arg\max} \frac{1}{N-1} \sum_{i=2}^{N}{a_{i,i-1}^{lh}}
$
for each layer, we must condition the current token's confidence score on the previous token's confidence score, or we may may assume that the current token is correct when it depends strongly on a previous hallucination.<br> For this, define the layer-wise confidence $c_l(y_i)$ on the confidence of the previous token $c_l(y_{i-1})$, the attention weight $a_{i,i-1}^{l,h_l}$ from the selected head, and the conditional probability of the current token $P(y_i|y_{<i}, x)$ as follows:
$$c_l(y_i) =
\begin{cases}
P(y_i \mid \mathbf{x}), & \text{if } i = 1, \\[6pt]
\alpha \cdot P(y_i \mid y_{<i}, \mathbf{x}) 
+ (1 - \alpha)\, a^{\,l,h_l}_{i,i-1}\, c_l(y_{i-1}), & \text{if } i > 1,
\end{cases}
$$
where $\alpha$ balances the contribution of each component. The paper finds that $\alpha=0$ works best for summarization, while $\alpha=0.2$ works best for QA and general generation.

## 3. Aggregating across layers and computing uncertainty score

In general, sequence-level errors are distributed across all tokens, like in summarization, or localized in a single fact-related token, like in QA. To take both into account, we compute the final confidence score as:
$$u_l(y) = -\frac{1}{N} \sum_{i=1}^{N} \log c_l(y_i)$$
and aggregate layer-wise by simply taking the max:
$$u(y) = \max_{l \in \{L\}} u_l(y)$$


## 4. Extra (not in paper): finding the threshold for flagging hallucinations

Option 1: don't use a threshold, just rank the outputs by uncertainty score and use the top-k most uncertain ones as CoT triggers:
+: generalizes well, no need for calibration, flexible, allows for task-specific triggers.
-: need to wait for generation to finish to rank, defeats the purpose of fast and budget-wise rollback, hard to interpret, no clear cutoff, how do I show the clear interest of rauq-minicot.

Option 2: train on varied datasets (QA, general, summarization, translation) with known hallucinations to find a threshold that maximizes F1 score:
+: interpretable, clear cutoff, easy to show the interest of rauq-minicot.
-: needs threshold calibration, may not generalize well to new tasks, brittle.