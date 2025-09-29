Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.001.png) Quantification for LLMs![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.002.png)

Artem Vazhentsev1,2 Lyudmila Rvanova2,4 Gleb Kuzmin2,4 Ekaterina Fadeeva6 Ivan Lazichny2 Alexander Panchenko1,2 Maxim Panov3 Timothy Baldwin3,5     Mrinmaya Sachan6 Preslav Nakov3 Artem Shelmanov3

1Skoltech 2AIRI 3MBZUAI

4FRC CSC RAS 5The University of Melbourne 6ETH Zürich

<vazhentsev@airi.net> <artem.shelmanov@mbzuai.ac.ae>

Abstract

Large language models (LLMs) exhibit impressive fluency, but often produce crit- ical errors known as “hallucinations”. Uncertainty quantification (UQ) methods are a promising tool for coping with this fundamental shortcoming. Yet, existing UQ methods face challenges such as high computational overhead or reliance on supervised learning. Here, we aim to bridge this gap. In particular, we propose RAUQ (Recurrent Attention-based Uncertainty Quantification), an unsupervised approach that leverages intrinsic attention patterns in transformers to detect hal- lucinations efficiently. By analyzing attention weights, we identified a peculiar pattern: drops in attention to preceding tokens are systematically observed during incorrect generations for certain “uncertainty-aware” heads. RAUQ automatically selects such heads, recurrently aggregates their attention weights and token-level confidences, and computes sequence-level uncertainty scores in a single forward pass. Experiments across 4 LLMs and 12 question answering, summarization, and translation tasks demonstrate that RAUQ yields excellent results, outperforming state-of-the-art UQ methods using minimal computational overhead (<1% latency). Moreover, it requires no task-specific labels and no careful hyperparameter tuning, offering plug-and-play real-time hallucination detection in white-box LLMs.

1 Introduction

Large language models have become the de facto backbone of modern NLP systems; yet, the impres- sive fluency of their responses often conceals various inconsistencies known as “hallucinations” [\[24](#_page10_x108.00_y551.78)]. There are several ways to address hallucinations, such as post-hoc verification using external knowl- edge bases [\[33](#_page11_x108.00_y295.08)], incorporating retrieval-augmented generation to ground outputs in factual data [\[28](#_page11_x108.00_y71.05)], or filtering/altering responses based on the uncertainty of a model [\[27,](#_page10_x108.00_y687.27)[ 14](#_page10_x108.00_y71.05)]. The latter approach, based on uncertainty, is the focus of this work.

Uncertainty is a fundamental concept in machine learning, reflecting the fact that we usually lack complete information about the model’s predictions or parameters [[16, ](#_page10_x108.00_y143.20)[23, ](#_page10_x108.00_y521.16)[25\].](#_page10_x108.00_y593.31) High predictive uncertainty typically signals a greater likelihood of hallucinations in the model output. Unlike verification methods that rely on external knowledge sources to detect hallucinations, uncertainty quantification (UQ) leverages the model’s internal capabilities, thereby mitigating issues related to the completeness of external sources and offering greater versatility. As shown in previous work, uncertainty scores can be used to detect hallucinations that arise due to limitations of LLM parametric knowledge or due to the ambiguity of requests in various generation tasks [[32, ](#_page11_x108.00_y254.64)[17, ](#_page10_x108.00_y195.64)[2\],](#_page9_x112.98_y129.51) including question-answering, machine translation, text summarization, and speech recognition.

Preprint. Under review.

UQ for classification and regression tasks is a well-established area spanning decades of research [\[54, ](#_page13_x108.00_y122.66)[19,](#_page10_x108.00_y311.42)[ 48,](#_page12_x108.00_y482.04)[ 46,](#_page12_x108.00_y384.87)[ 43,](#_page12_x108.00_y190.04)[ 21](#_page10_x108.00_y427.20)]. At the same time, UQ for generative tasks has only recently emerged as an active topic and still features open challenges. A crucial difference over classification is that an LLM performs not a single, but multiple conditionally dependent predictions. While recent work has proposed several promising techniques for quantifying predictive uncertainty in generation, e.g. [\[27, ](#_page10_x108.00_y687.27)[14,](#_page10_x108.00_y71.05)[ 10,](#_page9_x108.00_y554.00)[ 35,](#_page11_x108.00_y419.61)[ 31](#_page11_x108.00_y214.20)], prior methods have limitations. Namely, information–based scores such as maximum sequence probability (MSP) and token-level entropy are simple and fast, but often underperform on long-form generation tasks [[52,](#_page12_x108.00_y676.36) [44\].](#_page12_x108.00_y254.98) Sampling-based scores offer stronger performance, but incur large computational overhead [[27,](#_page10_x108.00_y687.27) [31,](#_page11_x108.00_y214.20) [42\].](#_page12_x108.00_y136.00) Supervised confidence regressors [[1,](#_page9_x112.98_y88.72) [6\],](#_page9_x112.98_y358.12) i.e., thin supplementary modules trained on supervised annotation, yield accurate scores, but require costly, task–specific annotation and often fail to generalize to out-of-distribution data or across tasks [[44\]. ](#_page12_x108.00_y254.98)Thus, despite the recent surge of developments of UQ for LLMs, the research community still lacks an effective, versatile UQ method that (i) avoids the high computational costs associated with sampling-based approaches, and (ii) is robust across tasks and domains.

In this work, we aim to construct such a method. For this purpose, we peek into the attention weights of the transformer and identify patterns that are highly indicative of the presence of hallucinations. Self-attention matrices encode how strongly each newly generated token attends to its immediate context. We empirically observe a systematic drop in the attention weight to the preceding tokens in specific attention heads precisely at positions where the model later proves to be factually incorrect (Figure[ 1).](#_page2_x108.00_y66.54) Based on this finding, we argue that a small number of attention heads capture the behavior of transformer-based LLMs under uncertainty. We propose a method that automatically identifies such “uncertainty-aware” heads inside individual LLM layers and extracts the token-level signal from them. The method recurrently fuses this signal with token probabilities and confidence scores from previously generated tokens, capturing the conditional dependencies across generation steps. Finally, it aggregates token-level scores across the generated sequence and layers. The resulting sequence- level uncertainty score achieves state-of-the-art performance and demonstrates high robustness to the choice of its single hyperparameter. Moreover, since attention weights are readily available at inference time for white-box LLMs, the method requires no additional generation passes and adds almost no computational overhead to response latency.

Contributions:

1. In-depth analysis of attention-based patterns in LLMs associated with hallucinations, which uncovers what we term “uncertainty-aware” heads, i.e., attention heads whose signals notably correlate with hallucination occurrences.
1. RAUQ (Recurrent Attention-based Uncertainty Quantification) – an unsupervised UQ method that turns raw attentions and LLM probabilities into reliable uncertainty scores while adding only <1% latency. RAUQ requires no task-specific labels or tuning of hyperpa- rameters for a particular LLM, making it an easy plug-and-play for white-box LLMs.
1. Thorough experimental evaluation on four LLMs and 12 benchmarks, spanning summa- rization, translation, and question answering, showing that RAUQ achieves state-of-the-art results over 15 baselines. We also demonstrate the importance of each component within the method and illustrate that each individually could improve other UQ methods.

2 Related Work

Several recent studies have proposed attention-based UQ methods for detecting hallucinations in LLM-generated outputs.

Zhang et al. [\[53\]](#_page13_x108.00_y71.05) use attention weights to propagate uncertainty across generation steps by capturing conditional dependencies, helping to mitigate overconfidence from prior hallucinations. However, attention plays a secondary role, with the method mainly relying on probability and entropy.

Yuksekgonul et al. [\[50\]](#_page12_x108.00_y579.20) perform a mechanistic investigation of attention patterns linked to LLM factual errors and propose a supervised UQ method called SAT Probe. They associate hallucinations withweakattentiontoso-called“constrained”tokensintheprompt–keypromptelementsthatnarrow down the scope of the answer. However, their experiments show that SAT Probe performs only on par with or slightly better than baselines. In a similar vein, Contextualized Sequence Likelihood [\[30\] ](#_page11_x108.00_y162.85)leverages attention to important tokens in the input context to reweight the contribution of token logits when computing weighted sequence likelihood. Lookback Lens [\[8\]](#_page9_x112.98_y450.60) leverages attention maps to

Question: What is King Henry holding in the Portrait of Henry VIII? ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.003.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.004.png)

0\.7![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.005.jpeg)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.006.png)

` `King

` `Henry

0\.6

` `is

` `holding

0\.5

` `a

` `fal

0\.4

con

` `in

0\.3

` `the

` `Portrait

` `of 0.2  Henry

0\.1

` `VIII

.

0 5 10 15 20 25 30

Attention Head

<a name="_page2_x108.00_y66.54"></a>Figure 1: Attention weights in the 29th layer of Llama 3.1 8B from each generated token to its preceding token, given the prompt What is King Henry holding in the Portrait of Henry VII?. The y axis specifies the generated tokens, and the x axis specifies the attention heads. Warmer colors indicate higher attention values. The output contains the factually incorrect token falcon (the correct answer is glovesand dagger). Notably, the 25th attention head stands out by consistently assigning relatively high attention to the preceding token. However, for the hallucinated token falcon, this attention drops sharply – potentially serving as a signal for hallucination detection.

construct features for a supervised hallucination detector. The authors hypothesize that hallucinations correlate with less attention paid to the input context. They compute the ratio between cumulative attention weights to tokens in the answer and the prompt and train a linear classifier on top of these features. Attention-based features are also used in Trainable Attention-Based Dependency [\[44](#_page12_x108.00_y254.98)]. This method adds recurrence when computing uncertainty for subsequent tokens. It demonstrates strong results for in-domain tasks, outperforming Lookback Lens, but both methods lack generalization due to their supervised nature.

Finally, Sriramanan et al. [\[40\]](#_page11_x108.00_y665.46) recently proposed the Attention Score method, where they compute a length-normalized sum of log attention weights to preceding tokens across the prompt and the answer. Lower scores signal the presence of hallucination.

Although recent studies show that attention weights offer valuable signals for detecting hallucinations in LLM outputs, existing methods suffer from various limitations that hinder their effectiveness. SAT Probe, Lookback Lens, and TAD are supervised and show limited generalization beyond their training domain. Zhang et al. [\[53\]](#_page13_x108.00_y71.05) and Lin et al. [\[30\]](#_page11_x108.00_y162.85) leverage attention only as a supplement to other scores. Sriramanan et al. [\[40\]](#_page11_x108.00_y665.46) do not select proper attention heads before averaging, and allow the attention weights from prompt tokens to participate in the aggregation for the final score, which causes underperformance.

In this work, we aim to overcome the limitations of existing methods. To this end, we identify strong and generalizable attention-based patterns for LLM hallucination detection, isolate the key techniques required to effectively exploit these patterns, and develop a robust unsupervised UQ method that achieves state-of-the-art performance.

3 Identifying Hallucination-Associated Patterns in Attention Maps

We analyze the model’s attention maps when an LLM generates correct vs. incorrect outputs. We start with an analysis of attention weights to the immediately preceding token, i.e. ali,ih −1 – attention

weight to the {i − 1}-th token during the generation of i-th token from the layer l and attention head h. Let N be the number of generated tokens in the answer, H the number of attention heads in each layer, and L be the number of layers in the LLM. For illustration, we use the[ Llama 3.1 8B ](https://huggingface.co/meta-llama/Llama-3.1-8B)model.

Difference between attention weights for hallucinated and non-hallucinated tokens. Figure[ 1 ](#_page2_x108.00_y66.54)presents an example of the attention weights to preceding tokens ali,ih −1 in one of the LLM layers

for the input question from the TruthfulQA dataset: What is King Henry holding in the Portrait of Henry VII? Most of the generated tokens are aligned with the question. However, the token falcon represents a hallucination, i.e. it is factually incorrect (the answer should be glove and dagger).

5

Correct ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.007.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.008.png)

0 0.6 ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.009.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.010.png)2 0.5

0\.4

4

0\.3 6 0.2

8 0.1 0.0

0 5 10 15 20 25 30

0 ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.011.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.012.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.013.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.014.png)

0\.20

2

0\.15

4

0\.10

6

8 0.05 0.00

0 5 10 15 20 25 30

Attention Head

Incorrect 

0 0.6 ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.015.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.010.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.016.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.017.png)2 0.5

0\.4

4

0\.3 6 0.2

8 0.1 0.0

0 5 10 15 20 25 30

0 ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.018.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.019.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.020.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.014.png)

0\.20

2

0\.15

4

0\.10

6

8 0.05

0\.00 0 5 10 15 20 25 30

Attention Head



<a name="_page3_x108.00_y66.54"></a>Figure 2: Average attention weights to the preceding token, aggregated over all answer tokens for questions from the TruthfulQA dataset using Llama 3.1 8B. The top 10 highest- and lowest-quality answers, as determined by a quality metric, are labeled as correct and incorrect, respectively. The black dashed box highlights the head with the highest average attention.

For most attention heads, the weights to previous tokens remain low across all generated tokens. In contrast, the 25th head exhibits a distinct pattern: it assigns relatively high attention to the preceding token for non-hallucinated (i.e., correct) tokens, but this attention drops significantly for the hallucinated token falcon.

This example demonstrates that attention weights from a small subset of attention heads can notably correlate with the factual correctness of generated tokens. While the choice of layer and head might vary, this case suggests that certain heads in specific layers are “uncertainty-aware”, i.e., they are sensitive to generation accuracy and could help to identify hallucinations. More examples of the similar pattern for Llama and other LLMs are presented in Figures[ 6 ](#_page20_x108.00_y124.64)to[ 9 ](#_page21_x108.00_y446.79)in Appendix[ F.](#_page20_x108.00_y72.00)

Difference between average attention weights for incorrect and correct answers. We begin by selecting 10 correct and 10 incorrect answers generated by the LLM. To evaluate the correctness of each answer, we use AlignScore – a continuous metric that quantifies semantic similarity between the generated response and the gold-standard answer [[51](#_page12_x108.00_y622.33)]. We sort all generations by their AlignScore, and designate the top 10 as correct answers and the bottom 10 as incorrect.

Then, we compute the average attention weight to the previous token across all tokens in the answer using the attention heads in the 29th and 22nd layers of the LLM, i.e. a¯lh = N1−1 Ni=2 ali,ih −1.

Figure[ 2 ](#_page3_x108.00_y66.54)presents the resulting values, where each row corresponds to a single selected answer, and each column indicates the average attention weight from a specific head.

The attention maps in the figure demonstrate that certain heads consistently assign higher average attention when the LLM generates correct answers as compared to incorrect ones. Moreover, there is a notable correlation between the quality of the answer and average attention (see Figure [3b).](#_page4_x108.00_y66.54) This way, we empirically discovered a pattern for assessing the correctness of LLM generations.

Should we select uncertainty-aware heads, and how should we do it? We compute the average attention score a¯lh across tokens in two scenarios: (1) attention values are averaged across all heads

in a layer, i.e. a¯l = H a¯lh; (2) attention values are extracted from a single head with the highest

h=1

average attention across tokens, i.e. a¯lhl , where hl = argmaxh=1...H a¯lh. Figure 3a[ compares](#_page4_x108.00_y66.54) the resulting values for correct and incorrect answers.

When using only the selected attention head, we observe a clear difference in the values between correctand incorrectanswers. However, averaging attentionacrossall headseliminatesthisdifference. This once again highlights the importance of focusing on specific uncertainty-aware heads. These heads can be identified by selecting those with the highest average attention weights across all tokens.

Do we need to look further back at preceding tokens to better detect hallucinations? We analyze the attention weights to multiple preceding tokens. Here, we compute alh – an attention weight to

i,i−k

the {i − k}-th token (k-th preceding token), k = 1,..., 6. Figure [4 ](#_page4_x108.00_y241.65)shows the difference between the average attention weights of the correct and incorrect answers.

We see that the attention weights differ substantially between correct and incorrect answers only for the two preceding tokens, with almost zero differences observed for earlier tokens. Notably, the difference is substantially larger for the first preceding token as compared to the second one.

Mean Attention From the Selected Head![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.021.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.022.png)

Selected Head

Average Over Heads 0.65 0.60

|||||||||||||||
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
|||1|1\.7%|||||||||||
|||||||||||||||
||||||3\.0|%||||||||
1

1

0\.55

0\.50 0.45 0.40 0.35 0.30

0

0

0\.39 0.40 0.41 0.42 0.43 0.030 0.031 0.032 0.033 0.034 0.035

Mean Attention Mean Attention

0\. Incorrect Answer 1. Correct Answer 0.2 0.3 0.4 0.5 0.6

Attention Threshold

<a name="_page4_x108.00_y66.54"></a>(a) (b)

Figure 3: Attention weights to the preceding token averaged across all tokens in the generated responses of Llama 3.1 8B on the TruthfulQA dataset. a): Comparison between incorrect (AlignScore < 0.1) and correct (AlignScore > 0.9) answers. Attention values are presented for two scenarios: (left) from the selected head with the highest average attention; (right) averaged across all heads. b): The relationship between average response quality and the average attention weight in the selected head.

0\.04 0.03 0.02 0.01 0.00

||||||||
| :- | :- | :- | :- | :- | :- | :- |
||||||||
i-6 token i-5 token i-4 token i-3 token i-2 token i-1 token

Mean Attention (Correct - Incorrect)

<a name="_page4_x108.00_y241.65"></a>Figure 4: Difference between correct (AlignScore > 0.9) and incorrect answers (AlignScore < 0.1) in average attention weights to preceding tokens during the generation of answers for the questions from the TruthfulQA dataset using Llama 3.1 8B.

Summary. Our analysis uncovers attention patterns associated with the factuality of individual tokens and LLM responses in general. A key observation is that such systematic patterns emerge only for a small subset of specific attention heads. Effectively leveraging them requires first identifying the relevant uncertainty-aware attention heads. We also observe that the immediately preceding token provides the strongest signal, leading us to focus solely on it in our method design and subsequent experiments. Below, we leverage the insights from this mechanistic investigation to develop a new unsupervised UQ method for LLMs.

4 RAUQ: Recurrent Attention-Based Uncertainty Quantification Method

Let x be the input sequence and y = y1y2 ...yN be its corresponding output sequence of length N.

Selecting an attention head in each layer. For an LLM with L layers and H attention heads per layer, we first select the most informative head. For each layerl, we select the head with the maximum average attention weights between consecutive tokens:

1

l h=1...H N − 1 Ni=2 ali,ih −1, (1) h (y) = argmax

where alh is the attention weight from token yi to yi−1 computed by the h-th head in the layer l.

i,i−1

Token-level layer-wise recurrent confidence score. Following [\[53](#_page13_x108.00_y71.05)], we acknowledge that computing uncertainty at the generation step i requires propagating uncertainty from previous steps, due to the conditional dependencies in the probability distribution modeled by the LLM. Namely, even if previous tokens were generated with high uncertainty, a model may condition on them and be highly confident in a current token prediction. To take into account this issue, we introduce a formulation that recurrently incorporates uncertainty from previous steps. We recurrently compute the confidence score cl(yi) for the i-th token by leveraging the confidence of the previous token cl(yi−1), the

lhl

attention weight ai,i−1 from the selected head hl = hl(y), and the conditional probability of the current token P (yi | y<i,x) as follows:

<a name="_page4_x166.47_y701.62"></a>l i Pα (·yPi (|yxi )|,y<i,x) + (1 − α) ·ali,ih−l 1 ·cl(yi−1), ifif ii => 11,, (2) c (y ) =

Algorithm 1: RAUQ: Recurrent Attention-based Uncertainty Quantification method![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.023.png)

<a name="_page5_x108.00_y91.58"></a>Data: Input prompt x, LLM generation y = y1:N , LLM attention weights ali,ih −1 for each layer l![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.024.png)

and each head h, token probabilities P (yi | y<i,x) and a hyperparameter α.

Result: Uncertainty score u(y)

// Selection of uncertainty-aware heads

1 for l ← 1 to L do

1  N![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.025.png)

lh

2  hl ← argmaxh=1...H N − 1 i=2 ai,i−1;

// Computing token-level confidence scores with uncertainty-aware heads

3 for i ← 1 to N do ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.026.png)4 if i == 1 then![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.027.png)

5 cl(yi) ← P (yi | x); 6 else

lh

7 cl(yi) ← αP (yi | y<i,x) + (1 − α) ai,il−1 cl(yi−1);

// Computing layer-wise and final uncertainty scores

N

8 ul(y) ← − N1 i=1 logcl(yi);

9 u(y) ← maxl∈L ul(y);

10 return u(y);![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.028.png)

where α is a hyperparameter that balances the contributions of each component. This recurrent formulation also helps to avoid an explosion in confidence scores with an increase in sequence length. We present an ablation study with the impact of varying the parameter α in Section[ 5.3 ](#_page7_x108.00_y417.90)and show that a single value provides robust performance across various tasks and even models.

Sequence-level layer-wise uncertainty score. Sequence-level errors are typically either (1) dis- tributed across all tokens, e.g. in the summarization task; or (2) localized in a single fact-related token, e.g. in the QA task. To take into account both cases in the sequence-level uncertainty score, we compute the mean logarithm of the confidence scores across all tokens in the reply (importantly, we do not aggregate scores for tokens in the prompt):

1 N

ul(y) = − N i=1 logcl(yi).

(3)

Final uncertainty score. Finally, to aggregate the layer-wise uncertainty scores in an unsupervised manner, we compute the maximum uncertainty score across the set of layers:

u(y) = max ul(y), (4)

l∈L

where L denotes the set of the most informative layers. Following previous work [[44\],](#_page12_x108.00_y254.98) we select these layers from the middle of the model. An ablation study with various aggregation functions is presented in Section[ 5.3.](#_page7_x108.00_y417.90) The step-by-step description of RAUQ is presented in Algorithm[ 1.](#_page5_x108.00_y91.58)

5 Experiments

1. Experimental Setup

We conducted extensive experiments across three key generation tasks: question answering (“QA”), abstractive text summarization (“Summ”), and machine translation (“MT”). For each task, we evaluated RAUQ’s ability to identify and filter out unreliable output through selective generation. We set α = 0.0 for the summarization task and α = 0.2 for all other tasks.

Datasets. For QA, we use seven datasets: TruthfulQA [\[29](#_page11_x108.00_y111.50)], SciQ [\[47\]](#_page12_x108.00_y428.00) for scientific QA, MMLU [\[22](#_page10_x108.00_y479.64)], TriviaQA [[26\]](#_page10_x108.00_y623.93) for trivia questions, CoQA [[36\]](#_page11_x108.00_y481.87) for conversational QA, MedQUAD [[4\]](#_page9_x112.98_y254.72) for medical questions, and GSM8k [\[9\]](#_page9_x112.98_y513.21) for mathematical reasoning. For summarization, we use three datasets with different summarization types: CNN/DailyMail [[39\]](#_page11_x108.00_y614.10) for news article summarization, SamSum [[18\] ](#_page10_x108.00_y258.98)for dialogue summarization, and XSum [[34\]](#_page11_x108.00_y357.35) for summarizing into a single sentence. For the MT <a name="_page6_x108.00_y66.54"></a>UQ Method Llama-3.1 8B Qwen-2.5 7B Gemma-2 9B Falcon-3 10B~~ Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.029.png)

QA Summ MT QA Summ MT QA Summ MT QA Summ MT

MSP .347 .129 .397 .329 .350 .369 .361 .176 .381 .345 .174 .333 .307 Perplexity .347 −.311 .380 .343 −.129 .406 .383 −.296 .405 .356 −.146 .439 .181 CCP .285 .148 .340 .271 .246 .327 .329 .147 .320 .299 .066 .287 .255 Attention Score .014 .053 .178 .038 .031 .142 .064 .120 .146 .054 −.091 .089 .070 Focus .320 −.110 .361 .264 .087 .380 .416 −.085 .385 .313 −.007 .362 .224 Simple Focus .342 .056 .415 .342 .252 .399 .396 .067 .422 .351 −.019 .385 .284



|<p>DegMat NLI Score entail. Ecc. NLI Score entail. EVL NLI Score entail. Lexical Similarity Rouge-L EigenScore</p><p>LUQ</p><p>Semantic Entropy</p><p>SAR</p><p>Semantic Density</p>|.306|.199|.239|.356|.183|.275|.337|.150|.259|.352|.088|.222|.247|
| :- | - | - | - | - | - | - | - | - | - | - | - | - | - |
||.274 .293 .250 .232 .287|.029 .188 .133 .082 .168|.284 .217 .324 .285 .214|.322 .349 .334 .298 .351|.039 .181 .141 .007 .047|.306 .245 .327 .302 .213|.298 .332 .306 .267 .344|−.018 .139 .132 .119 .206|.290 .252 .342 .226 .259|.327 .351 .285 .247 .335|.044 .140 .033 .023 .142|.281 .206 .275 .236 .196|.206 .241 .240 .194 .230|
||.254 .310|.076 .238|.315 .370|.281 .351|.292 .112|.317 .393|.291 .361|.154 .159|.337 .414|.320 .334|.083 .033|.291 .337|.251 .284|
||.330|.043|.264|.352|.095|.291|.375|.055|.255|.358|.031|.280|.227|

RAUQ (Ours) .396 .375 .452 .358 .415 .438 .421 .471 .473 .392 .316 .465 .414![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.030.png)

Table 1: Mean PRR↑ across tasks for the evaluated LLMs. Warmer color indicates better results.

task, we evaluate on two language pairs from WMT: German–English from WMT19 [\[3\]](#_page9_x112.98_y170.30) and French– English from WMT14 [\[5](#_page9_x112.98_y284.60)]. Detailed statistics for all datasets are presented in Table[ 3 ](#_page14_x108.00_y135.55)in Appendix[ A.](#_page14_x108.00_y72.00)

Models. To show the generalization of the method across various models, we use several widely used open-weight LLMs: Llama-3.1 8B [[11\],](#_page9_x108.00_y616.61) Qwen-2.5 7B [[49\],](#_page12_x108.00_y546.98) Gemma-2 9B [[38\],](#_page11_x108.00_y573.66) and Falcon-3 10B [\[13\].](#_page9_x108.00_y709.09) Detailed descriptions of generation parameters are presented in Table[ 3 ](#_page14_x108.00_y135.55)in Appendix[ A.](#_page14_x108.00_y72.00)

Uncertainty quantification baselines. We compare the proposed RAUQ method with 15 diverse UQ baselines. As a sanity check, we include simple unsupervised baselines such as Maximum Sequence Probability (MSP) and Perplexity [[15\].](#_page10_x108.00_y101.67) Among state-of-the-art baselines for whitebox LLMs, we compare our method to Semantic Entropy [\[27](#_page10_x108.00_y687.27)], hallucination detection with stronger focus (“Focus”) [\[53](#_page13_x108.00_y71.05)], Claim-Conditioned Probability (“CCP”) [[12](#_page9_x108.00_y646.48)], EigenScore [\[7](#_page9_x112.98_y409.81)], Shifting Attention to Relevance (“SAR”) [\[10](#_page9_x108.00_y554.00)], Semantic Density [\[35](#_page11_x108.00_y419.61)], and Attention Score [\[40](#_page11_x108.00_y665.46)]. Additionally, we consider UQ methods for black-box LLMs, as they also demonstrate strong performance in recent works despite not having access to LLM logits or its hidden states. We use Lexical Similarity based on Rouge-L [\[15](#_page10_x108.00_y101.67)], Long-text Uncertainty Quantification (“LUQ”) [\[52](#_page12_x108.00_y676.36)], and methods from [\[31\]](#_page11_x108.00_y214.20) – Degree Matrix (“DegMat”), Eccentricity, and Sum of Eigenvalues of the graph Laplacian (“EVL”).

Evaluation metrics. As the main evaluation metric, we use the standard Prediction Rejection Ratio (PRR) [[32,](#_page11_x108.00_y254.64)[ 42](#_page12_x108.00_y136.00)]. PRR is calculated as the ratio of the area between the rejection curves of the UQ method and the random UQ baseline, and the same area between the ideal UQ method and the random UQ baseline. The rejection curve plots the average quality of remaining responses when we abstain from a fraction of the most uncertain predictions. We compute PRR over only the first 50% of the curve, as rejecting more than half of the instances is typically impractical. The metric is normalized so that a PRR of zero or below indicates performance at or below the level of random chance, while values approaching one reflect optimal performance. PRR is analogous to ROC-AUC or PR-AUC, but unlike them, it can be applied not only to discrete quality metrics (e.g. correct vs. incorrect answer) but also to continuous ones, such as those commonly used in summarization and MT. For different generation tasks, we use different response quality metrics: accuracy for MMLU and GSM8k; COMET [[37\]](#_page11_x108.00_y522.31) for MT; and AlignScore [[51\]](#_page12_x108.00_y622.33) for the rest. Additionally, we calculate ROC-AUC using discrete quality metrics obtained by thresholding the original continuous values.

2. Main Results

Table[ 1 ](#_page6_x108.00_y66.54)presents the mean PRR for each task (QA, Summ, and MT) for each of the evaluated LLMs. To compute the mean PRR for each task, we average the PRR scores across all relevant datasets, for example, XSum, CNN, and SamSum for summarization. These aggregated PRR scores provide a robust measure of the performance of various methods for each task and model. Detailed results for each model and dataset are presented in Tables [11 ](#_page18_x108.00_y124.64)to [14 ](#_page19_x108.00_y319.56)in Appendix [E. ](#_page18_x108.00_y72.00)The results using the ROC-AUC metric are presented in Table[ 8 ](#_page16_x108.00_y224.40)in Appendix[ D.1.](#_page16_x108.00_y91.11)

The results demonstrate that the proposed RAUQ method consistently outperforms previous state-of- the-art methods for the summarization and translation tasks by a substantial margin for all evaluated LLMs. For instance, for the summarization task using Gemma-2 9B, RAUQ largely outperforms the

XSUM SamSum CNN WMT14 WMT19 MedQUAD![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.031.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.032.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.033.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.034.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.035.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.036.png)

0\.3 0.40

0\.5 0.3 0.5 0.2

0\.35

0\.2

0\.0 0.4

0\.2 0.30 0.1

0\.1![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.037.png)

0\.5 0.3 0.0

0\.25

1\.0 0.1 0.20 0.2 0.1

0\.0

0\.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0

![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.038.png) ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.039.png) ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.040.png) ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.041.png) ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.042.png) ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.043.png)

TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.044.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.045.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.046.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.047.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.048.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.049.png)

0\.5 0.65

0\.5

3. 0.25 0.30

   4. 0.60

0\.20 0.4

0\.10.2 0.15 0.3 0.55 0.25

0\.3

0\.0 0.10 0.2 0.50 0.20

0\.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0

RAUQ (ours) MSP Selected ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.050.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.051.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.052.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.053.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.054.png)

<a name="_page7_x108.00_y66.54"></a>Figure 5: PRR↑ as a function of the hyperparameter α for Llama 3.1 8B. The vertical line marks the value of α used in our experiments.

second-best method (LUQ) by 0.265 of PRR. In contrast, other single-generation methods based on the attention weights, such as Focus and Attention Score, perform close to random chance.

For the QA task, RAUQ also achieves the best results across all models often with a substantial margin over the second best method. Notably, RAUQ improves over the second-best method (MSP) for Llama-3.1 8B by 0.049 in terms of PRR. However, for Qwen-2.5 7B in the QA task, computationally intensive DegMat comes close, trailing RAUQ by just 0.002 PRR. However, RAUQ consistently outperforms all other sampling-based baselines on average.

Overall, while methods such as MSP, Focus, or SAR might achieve top performance in specific settings, RAUQ demonstrates the most robust performance across all tasks and models, consistently ranking as the best method by average performance in a task.

Tables[ 9 ](#_page17_x108.00_y166.83)and [10 ](#_page17_x108.00_y495.13)in Appendix [D.2 ](#_page16_x108.00_y423.19)also provide a comparison with supervised UQ methods. While RAUQ slightly underperforms compared to supervised methods on their in-domain data, it greatly outperforms them on average in out-of-domain scenarios.

3. Hyperparameter<a name="_page7_x108.00_y417.90"></a> Sensitivity and Ablation Studies

Impact of the hyperparameter α. The hyperparameter α from Equation [(2)](#_page4_x166.47_y701.62) balances the contribu- tions of attention, confidence from the previous token, and the conditional probability of the current token. When α is equal to 1, RAUQ becomes equivalent to perplexity. When α approaches 0, RAUQ relies solely on the attention weights from the selected head. Figure[ 5 ](#_page7_x108.00_y66.54)presents the impact of α on the performance of the RAUQ method for Llama 8b v3.1. For the summarization, setting α equal to 0 consistently yields the best results. For all other tasks, except MMLU, the best possible performance is achieved with α between 0.2 and 0.5.

While dataset-specific fine-tuning of α can lead to further improvements, we do not perform such careful tuning in our main experiments (Table [1).](#_page6_x108.00_y66.54) Instead, we select α using a small out-of-domain subset for Llama 8b v3.1 and apply this value uniformly across all datasets and LLMs. Despite this, RAUQ achieves consistently strong performance across diverse tasks and LLMs, often achieving the top or near-top results. Strong performance with a fixed hyperparameter underscores the robustness of the proposed method.

Aggregation functions. Table[ 5 ](#_page15_x108.00_y95.33)compares the performance of the RAUQ method using various aggregation functions of token-level confidence scores. We experiment with four aggregation strategies: mean, median, sum of logarithms (inspired by MSP), and mean of logarithms (inspired by perplexity). For the Summ tasks and certain QA datasets (SciQ, TriviaQA, and GSM8k), mean aggregationyieldsthe bestperformance. For MMLU,the sumoflogarithms substantially outperforms other aggregation strategies, while median performs best for XSum. However, the top two performing methods are those that apply length normalization. Among them, the mean of logarithms of token- level confidence scores used in RAUQ consistently delivers the strongest results across datasets.

Table [6 ](#_page15_x108.00_y188.26)compares the performance of RAUQ using various aggregation functions of layer-wise uncertainty scores. We consider three aggregation strategies: mean, median, and maximum. Both maximum and median yield similarly strong performance, while the mean aggregation performs

<a name="_page8_x108.00_y66.54"></a>UQ Method XSum SamSum CNN WMT14 WMT19 MedQUAD TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.055.png)

Attention Score .100 .017 .043 .176 .179 -.295 .081 -.028 -.142 .067 .209 .209 .051 Attention Score (Gen. Tokens) .595 .269 .278 .196 .198 -.305 -.020 .064 .124 .130 .232 .192 .163 Attention Score (Gen. Tokens, Selected Head) .547 .271 .300 .187 .200 -.113 -.025 .092 .161 .151 .414 .197 .198 RAUQ .566 .269 .290 .394 .509 .241 .364 .265 .506 .522 .549 .323 .400

Table 2: PRR↑ for Llama 3.1 8B across various modifications of the Attention Score method incorporating components from RAUQ. The best method is in bold, the second best is underlined.

slightly worse. Although the median slightly outperforms the maximum by an average margin of 0.003 PRR across tasks, this difference is negligible. Given that the maximum is a more intuitive choice – it effectively captures the peak uncertainty within a generation and achieves better results in 7 out of 12 tasks, we adopt it as the default layer-wise aggregation method in our experiments.

Recurrent uncertainty propagation functions. Table[ 7 ](#_page15_x108.00_y269.80)presents the performance of the RAUQ method using various recurrent formulas for the calculation of token-level confidence scores. We con- sider five modifications of Equation [(2):](#_page4_x166.47_y701.62) (1) removing attention weights, (2) removing recurrence, (3) replacing the confidence score of the previous token with its probability, (4) multiplying probabilities with attentions, and (5) the recurrent formula proposed in RAUQ.

The proposed formula achieves the best results on the majority of the datasets. Removing either recurrence or attention often leads to substantially worse performance. The results highlight the importance of each component in the proposed formula for achieving good results.

Extending our findings to the Attention Score method. To demonstrate the robustness and generalization of RAUQ components, we integrated them into the recently published Attention Score method [[40](#_page11_x108.00_y665.46)], resulting in two modifications. We compare (1) the original official implementation of Attention Score; (2) Attention Score that uses only the attention weights of the generated tokens, excluding the prompt; (3) Attention Score that combines the previous feature and implements also the selection of the uncertainty-aware attention heads; (4) the full RAUQ method with recurrence.

Results in Table [2 ](#_page8_x108.00_y66.54)show that excluding contributions from prompt tokens significantly boosts the average performance of Attention Score, yielding a 0.112 improvement in PRR. The highest improve- ment is achieved on the summarization tasks, where the modified Attention Score approaches the performance of RAUQ. Incorporating attention head selection further boosts the average performance by 0.035, with a large gain of 0.182 on MMLU. Nevertheless, our full method further incorporates to- ken probabilities and recurrently aggregates uncertainty scores from previous generation steps, which provides a distinct advantage. Overall, these results suggest that our findings regarding attention heads and design choices in RAUQ are systematic and generalize to prior UQ methods as well.

4. Computational Efficiency

To demonstrate the computational efficiency of RAUQ, we conducted a comprehensive runtime comparison against other state-of-the-art UQ methods using Llama 8b v3.1. All experiments were performed on a single 80GB NVIDIA H100 GPU using single-batch inference, following the same setup as in Table [1.](#_page6_x108.00_y66.54) Table [4 ](#_page14_x108.00_y340.43)in Appendix [B ](#_page14_x108.00_y307.51)reports the average runtime per instance for each UQ method, and quantifies their computational overhead relative to standard LLM inference without UQ.

State-of-the-art UQ methods such as DegMat [\[14](#_page10_x108.00_y71.05)], Semantic Entropy [\[27](#_page10_x108.00_y687.27)], and SAR [\[53\]](#_page13_x108.00_y71.05) introduce huge computational overhead (400–800%) due to repeated sampling from the LLM. In contrast, the RAUQ method introduces less than 1% overhead since it does not require sampling or inference of an auxiliary model, making it a fast, lightweight, and plug-and-play solution for any white-box LLM.

6 Conclusion

We introduced RAUQ, an unsupervised, attention-based framework that converts the intrinsic signals already produced by every transformer layer into reliable sequence-level uncertainty scores with a single forward pass. A simple head-selection heuristic, a recurrent confidence propagation rule, and a length-normalized aggregation allow RAUQ to capture both local spikes and global drifts in confidence without external supervision or sampling. Extensive experiments on 12 datasets spanning QA, Summ, and MT, and on 4 open-weight LLMs show that RAUQ delivers state-of-the-art performance with only <1 % latency overhead, making it a practical off-the-shelf UQ technique.

References

1. A.<a name="_page9_x112.98_y88.72"></a> Azaria and T. Mitchell. The internal state of an LLM knows when it‘s lying. In H. Bouamor,

   10. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967–976, Singapore, Dec. 2023. Association for Computational Linguistics.
1. J.<a name="_page9_x112.98_y129.51"></a> Baan, N. Daheim, E. Ilia, D. Ulmer, H.-S. Li, R. Fernández, B. Plank, R. Sennrich, C. Zerva, and W. Aziz. Uncertainty in natural language generation: From theory to applications. arXiv preprint arXiv:2307.15703, 2023.
1. L.<a name="_page9_x112.98_y170.30"></a> Barrault, O. Bojar, M. R. Costa-jussà, C. Federmann, M. Fishel, Y. Graham, B. Haddow,

   M. Huck, P. Koehn, S. Malmasi, C. Monz, M. Müller, S. Pal, M. Post, and M. Zampieri. Findings of the 2019 Conference on Machine Translation (WMT19). In O. Bojar, R. Chatterjee,

   C. Federmann, M. Fishel, Y. Graham, B. Haddow, M. Huck, A. J. Yepes, P. Koehn, A. Martins,

   C. Monz, M. Negri, A. Névéol, M. Neves, M. Post, M. Turchi, and K. Verspoor, editors, Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 1–61, Florence, Italy, Aug. 2019. Association for Computational Linguistics.

4. A.<a name="_page9_x112.98_y254.72"></a> Ben Abacha and D. Demner-Fushman. A question-entailment approach to question answer- ing. BMC Bioinform., 20(1):511:1–511:23, 2019.
4. O.Bojar,C.Buck,C.Federmann,B.Haddow,P.Koehn,J.Leveling,C.Monz,P.Pecina,M.Post,

   <a name="_page9_x112.98_y284.60"></a>H. Saint-Amand, R. Soricut, L. Specia, and A. Tamchyna. Findings of the 2014 workshop on statistical machine translation. In O. Bojar, C. Buck, C. Federmann, B. Haddow, P. Koehn,

   C. Monz, M. Post, and L. Specia, editors, Proceedings of the Ninth Workshop on Statistical Machine Translation, pages 12–58, Baltimore, Maryland, USA, June 2014. Association for Computational Linguistics.

6. S.<a name="_page9_x112.98_y358.12"></a> CH-Wang, B. Van Durme, J. Eisner, and C. Kedzie. Do androids know they’re only dreaming of electric sheep? In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 4401–4420, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
6. C.<a name="_page9_x112.98_y409.81"></a> Chen, K. Liu, Z. Chen, Y. Gu, Y. Wu, M. Tao, Z. Fu, and J. Ye. INSIDE: LLMs’ internal states retain the power of hallucination detection. In The Twelfth International Conference on Learning Representations, 2024.
6. Y.-S.<a name="_page9_x112.98_y450.60"></a> Chuang, L. Qiu, C.-Y. Hsieh, R. Krishna, Y. Kim, and J. R. Glass. Lookback lens: Detecting and mitigating contextual hallucinations in large language models using only attention maps. InY.Al-Onaizan,M.Bansal,andY.-N.Chen,editors,Proceedingsofthe2024Conference on Empirical Methods in Natural Language Processing, pages 1419–1436, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics.
6. K.<a name="_page9_x112.98_y513.21"></a> Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek,

   10. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021.
6. J.<a name="_page9_x108.00_y554.00"></a> Duan, H. Cheng, S. Wang, A. Zavalny, C. Wang, R. Xu, B. Kailkhura, and K. Xu. Shifting attention to relevance: Towards the predictive uncertainty quantification of free-form large language models. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5050–5063, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
6. A.<a name="_page9_x108.00_y616.61"></a> Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten,

   10. Yang, A. Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
6. E.<a name="_page9_x108.00_y646.48"></a> Fadeeva, A. Rubashevskii, A. Shelmanov, S. Petrakov, H. Li, H. Mubarak, E. Tsymbalov,

   G. Kuzmin, A. Panchenko, T. Baldwin, P. Nakov, and M. Panov. Fact-checking the output of large language models via token-level uncertainty quantification. In L.-W. Ku, A. Martins, and

   V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 9367–9385, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.

13. Falcon-LLM<a name="_page9_x108.00_y709.09"></a> Team. The falcon 3 family of open models, December 2024.
14. S.<a name="_page10_x108.00_y71.05"></a> Farquhar, J. Kossen, L. Kuhn, and Y. Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.
14. M.<a name="_page10_x108.00_y101.67"></a> Fomicheva, S. Sun, L. Yankovskaya, F. Blain, F. Guzmán, M. Fishel, N. Aletras, V. Chaud- hary, and L. Specia. Unsupervised quality estimation for neural machine translation. Transac- tions of the Association for Computational Linguistics, 8:539–555, 2020.
14. Y.<a name="_page10_x108.00_y143.20"></a> Gal and Z. Ghahramani. Dropout as a bayesian approximation: Representing model uncer- tainty in deep learning. In M. F. Balcan and K. Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1050–1059, New York, New York, USA, 20–22 Jun 2016. PMLR.
14. J.<a name="_page10_x108.00_y195.64"></a> Geng, F. Cai, Y. Wang, H. Koeppl, P. Nakov, and I. Gurevych. A survey of confidence estimation and calibration in large language models. In K. Duh, H. Gomez, and S. Bethard, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6577–6595, Mexico City, Mexico, June 2024. Association for Computational Linguistics.
14. B.<a name="_page10_x108.00_y258.98"></a> Gliwa, I. Mochol, M. Biesek, and A. Wawer. SAMSum corpus: A human-annotated dialogue dataset for abstractive summarization. In L. Wang, J. C. K. Cheung, G. Carenini, and F. Liu, editors, Proceedings of the 2nd Workshop on New Frontiers in Summarization, pages 70–79, Hong Kong, China, Nov. 2019. Association for Computational Linguistics.
14. J.<a name="_page10_x108.00_y311.42"></a> He, X. Zhang, S. Lei, Z. Chen, F. Chen, A. Alhamadani, B. Xiao, and C. Lu. Towards more accurate uncertainty estimation in text classification. In B. Webber, T. Cohn, Y. He, and

    25. Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8362–8372, Online, Nov. 2020. Association for Computational Linguistics.
14. J.<a name="_page10_x108.00_y374.76"></a> He, Y. Gong, Z. Lin, C. Wei, Y. Zhao, and K. Chen. LLM factoscope: Uncovering LLMs’ factual discernment through measuring inner states. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 10218– 10230, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
14. J.<a name="_page10_x108.00_y427.20"></a> He, L. Yu, S. Lei, C.-T. Lu, and F. Chen. Uncertainty estimation on sequential labeling via uncertainty transmission. In K. Duh, H. Gomez, and S. Bethard, editors, Findings of the Association for Computational Linguistics: NAACL 2024, pages 2823–2835, Mexico City, Mexico, June 2024. Association for Computational Linguistics.
14. D.<a name="_page10_x108.00_y479.64"></a> Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.
14. N.<a name="_page10_x108.00_y521.16"></a> Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel. Bayesian active learning for classifica- tion and preference learning. arXiv preprint arXiv:1112.5745, 2011.
14. L.<a name="_page10_x108.00_y551.78"></a> Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, 2025.
14. E.<a name="_page10_x108.00_y593.31"></a> Hüllermeier and W. Waegeman. Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. Machine learning, 110(3):457–506, 2021.
14. M.<a name="_page10_x108.00_y623.93"></a> Joshi, E. Choi, D. Weld, and L. Zettlemoyer. TriviaQA: A large scale distantly supervised challengedatasetforreadingcomprehension. InR.BarzilayandM.-Y.Kan,editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics.
14. L.<a name="_page10_x108.00_y687.27"></a> Kuhn, Y. Gal, and S. Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimationinnaturallanguagegeneration. InTheEleventhInternationalConferenceonLearning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.
28. P.<a name="_page11_x108.00_y71.05"></a> Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.
28. S.<a name="_page11_x108.00_y111.50"></a> Lin, J. Hilton, and O. Evans. TruthfulQA: Measuring how models mimic human falsehoods. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland, May 2022. Association for Computational Linguistics.
28. Z.<a name="_page11_x108.00_y162.85"></a> Lin, S. Trivedi, and J. Sun. Contextualized sequence likelihood: Enhanced confidence scores for natural language generation. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Pro- ceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10351–10368, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics.
28. Z.<a name="_page11_x108.00_y214.20"></a> Lin, S. Trivedi, and J. Sun. Generating with confidence: Uncertainty quantification for black-box large language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.
28. A.<a name="_page11_x108.00_y254.64"></a> Malinin and M. J. F. Gales. Uncertainty estimation in autoregressive structured prediction. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.
28. S.<a name="_page11_x108.00_y295.08"></a> Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. Koh, M. Iyyer, L. Zettlemoyer, and

    8. Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore, Dec. 2023. Association for Computational Linguistics.
28. S.<a name="_page11_x108.00_y357.35"></a> Narayan, S. B. Cohen, and M. Lapata. Don‘t give me the details, just the summary! topic- aware convolutional neural networks for extreme summarization. In E. Riloff, D. Chiang,

    8. Hockenmaier, and J. Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1797–1807, Brussels, Belgium, Oct.-Nov. 2018. Association for Computational Linguistics.
28. X.<a name="_page11_x108.00_y419.61"></a> Qiu and R. Miikkulainen. Semantic density: Uncertainty quantification for large language models through confidence measurement in semantic space. In A. Globerson, L. Mackey,

    500. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 134507–134533. Curran Associates, Inc., 2024.
28. S.<a name="_page11_x108.00_y481.87"></a> Reddy, D. Chen, and C. D. Manning. Coqa: A conversational question answering challenge. Transactions of the Association for Computational Linguistics, 7:249–266, 05 2019. ISSN 2307-387X.
28. R.Rei,C.Stewart,A.C.Farinha,andA.Lavie.<a name="_page11_x108.00_y522.31"></a> COMET:AneuralframeworkforMTevaluation. In B. Webber, T. Cohn, Y. He, and Y. Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2685–2702, Online, Nov. 2020. Association for Computational Linguistics.
28. M.<a name="_page11_x108.00_y573.66"></a> Rivière, S. Pathak, P. G. Sessa, C. Hardin, S. Bhupatiraju, L. Hussenot, T. Mesnard,

    8. Shahriari, A. Ramé, J. Ferret, et al. Gemma 2: Improving open language models at a practical size. CoRR, 2024.
28. A.<a name="_page11_x108.00_y614.10"></a> See, P. J. Liu, and C. D. Manning. Get to the point: Summarization with pointer-generator networks. In R. Barzilay and M.-Y. Kan, editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1073–1083, Vancouver, Canada, July 2017. Association for Computational Linguistics.
28. G.<a name="_page11_x108.00_y665.46"></a> Sriramanan, S. Bharti, V. S. Sadasivan, S. Saha, P. Kattakinda, and S. Feizi. Llm-check: Investigating detection of hallucinations in large language models. In A. Globerson, L. Mackey,

    D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 34188–34216. Curran Associates, Inc., 2024.

41. W.<a name="_page12_x108.00_y71.05"></a> Su, C. Wang, Q. Ai, Y. Hu, Z. Wu, Y. Zhou, and Y. Liu. Unsupervised real-time hallucination detection based on the internal states of large language models. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 14379–14391, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
41. R.<a name="_page12_x108.00_y136.00"></a> Vashurin, E. Fadeeva, A. Vazhentsev, L. Rvanova, D. Vasilev, A. Tsvigun, S. Petrakov,

    18. Xing, A. Sadallah, K. Grishchenkov, et al. Benchmarking uncertainty quantification methods forlargelanguagemodelswithlm-polygraph. TransactionsoftheAssociationforComputational Linguistics, 13:220–248, 2025.
41. A.<a name="_page12_x108.00_y190.04"></a> Vazhentsev, G. Kuzmin, A. Tsvigun, A. Panchenko, M. Panov, M. Burtsev, and A. Shel- manov. Hybrid uncertainty quantification for selective text classification in ambiguous tasks. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11659–11681, Toronto, Canada, jul 2023. Association for Computational Linguistics.
41. A.<a name="_page12_x108.00_y254.98"></a> Vazhentsev, E. Fadeeva, R. Xing, A. Panchenko, P. Nakov, T. Baldwin, M. Panov, and

    18. Shelmanov. Unconditional truthfulness: Learning conditional dependency for uncertainty quantification of large language models. arXiv preprint arXiv:2408.10692, 2024.
41. A.Vazhentsev,L.Rvanova,I.Lazichny,A.Panchenko,M.Panov,T.Baldwin,andA.Shelmanov.<a name="_page12_x108.00_y298.11"></a>Token-level density-based uncertainty quantification methods for eliciting truthfulness of large language models. In L. Chiruzzo, A. Ritter, and L. Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2246–2262, Albuquerque, New Mexico, Apr. 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6.
41. Y.<a name="_page12_x108.00_y384.87"></a> Wang, D. Beck, T. Baldwin, and K. Verspoor. Uncertainty estimation and reduction of pre-trained models for text regression. Transactions of the Association for Computational Linguistics, 10:680–696, 2022.
41. J.<a name="_page12_x108.00_y428.00"></a> Welbl, N. F. Liu, and M. Gardner. Crowdsourcing multiple choice science questions. In

    50. Derczynski, W. Xu, A. Ritter, and T. Baldwin, editors, Proceedings of the 3rd Workshop on Noisy User-generated Text, pages 94–106, Copenhagen, Denmark, Sept. 2017. Association for Computational Linguistics.
41. J.<a name="_page12_x108.00_y482.04"></a> Xin, R. Tang, Y. Yu, and J. Lin. The art of abstention: Selective prediction and error regularization for natural language processing. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1040–1051, Online, aug 2021. Association for Computational Linguistics.
41. A.<a name="_page12_x108.00_y546.98"></a> Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
41. M.<a name="_page12_x108.00_y579.20"></a> Yuksekgonul, V. Chandrasekaran, E. Jones, S. Gunasekar, R. Naik, H. Palangi, E. Kamar, and B. Nushi. Attention satisfies: A constraint-satisfaction lens on factual errors of language models. In The Twelfth International Conference on Learning Representations, 2024.
41. Y.<a name="_page12_x108.00_y622.33"></a> Zha, Y. Yang, R. Li, and Z. Hu. AlignScore: Evaluating factual consistency with a unified alignment function. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11328–11348, Toronto, Canada, July 2023. Association for Computational Linguistics.
41. C.<a name="_page12_x108.00_y676.36"></a> Zhang, F. Liu, M. Basaldella, and N. Collier. LUQ: Long-text uncertainty quantification for LLMs. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5244–5262, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics.
53. T.Zhang,L.Qiu,Q.Guo,C.Deng,Y.Zhang,Z.Zhang,C.Zhou,X.Wang,andL.Fu.<a name="_page13_x108.00_y71.05"></a> Enhancing uncertainty-based hallucination detection with stronger focus. In H. Bouamor, J. Pino, and K.Bali,editors, Proceedingsofthe2023ConferenceonEmpiricalMethodsinNaturalLanguage Processing, pages 915–932, Singapore, Dec. 2023. Association for Computational Linguistics.
53. X.<a name="_page13_x108.00_y122.66"></a> Zhang, F. Chen, C.-T. Lu, and N. Ramakrishnan. Mitigating uncertainty in document classification. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3126–3136, Minneapolis, Minnesota, jun 2019. Association for Computational Linguistics.

<a name="_page14_x108.00_y72.00"></a>A Dataset and Generation Statistics

The detailed description of the used datasets and the generation parameters of LLMs is presented in Table[ 3.](#_page14_x108.00_y135.55) For all LLMs, we used the same generation hyperparameters, while for each dataset, we separately fixed the number of few-shot and maximum generation length.

T<a name="_page14_x108.00_y135.55"></a>able 3: Statistics of the datasets and generation parameters of the used LLMs. For all datasets, we do not limit the maximum input length.![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.056.png)



|Dataset|Number of test samples|N-shot|Generation length|Do sample|Temperature|Top-p|Beams|
| - | - | - | :-: | - | - | - | - |

Task

Repetition Penalty

TruthfulQA 817 5 128![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.057.png)

SciQ 1000 0 20

MMLU 2000 5 3

QA TriviaQA 2000 5 20

all preceding

CoQA 2000 questions 20

False 1.0 1.0 1 1 MedQUAD 1000 5 128

GSM8k 1319 5 256

CNN/DailyMail 2000 0 128 ATS SamSum 819 0 128 XSum 2000 0 128

WMT19 (DE-EN) 2000 0 107

NMT

WMT14 (FR-EN) 2000 0 107

<a name="_page14_x108.00_y307.51"></a>B Computational Efficiency

T<a name="_page14_x108.00_y340.43"></a>able 4: Inference runtime of UQ methods measured on all test instances from all datasets with generations from Llama 8b v3.1. The best results are in bold.

Runtime![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.058.png)

Overhead UQ Method per batch

MSP 1.16±0.45 -

DegMat NLI Score Entail. 6.40±1.76 450% Lexical Similarity ROUGE-L 6.11±1.75 425% Semantic Entropy 6.40±1.76 450% SAR 10.71±3.21 820% Semantic Density 6.27±1.76 438%

RAUQ 1.17±0.45 0.3%

C Results of Ablation Studies

T<a name="_page15_x108.00_y95.33"></a>able 5: PRR↑ for Llama 8b v3.1 model for various aggregation function of token-level confidence scores. The best method is in bold, the second best is underlined.

Token Aggregation XSum SamSum CNN WMT14 WMT19 MedQUAD TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.059.png)

- 1 N ct (t )

−mediani=1Ni=1cl (iti) .571 .322 .306 .359 .485 .140 .304 .259 .511 .534 .526 .339 .388

N l

t .615 .261 .099 .249 .340 .154 .317 .234 .430 .432 .635 .253 .335 N t

- i=1 logcl (ti) .570 .270 .292 .224 .242 .107 .035 .114 .202 .300 .658 .213 .269
- N1 Ni=1 logctl (ti) .566 .269 .290 .394 .509 .249 .364 .265 .506 .522 .549 .323 .401

T<a name="_page15_x108.00_y188.26"></a>able 6: PRR↑ for Llama 8b v3.1 model for various aggregation function of layer-wise uncertainty scores. The best method is in bold, the second best is underlined.

Layer Aggregation XSum SamSum CNN WMT14 WMT19 MedQUAD TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.060.png)

|L|1 l∈L ul(y) .583 .273 .290 .389 .519 .154 .345 .274 .496 .535 .529 .337 .394 medianl∈L ul(y) .606 .263 .286 .388 .526 .246 .351 .267 .502 .532 .532 .340 .403 maxl∈L ul(y) .566 .269 .290 .394 .509 .249 .364 .265 .506 .522 .549 .323 .401

T<a name="_page15_x108.00_y269.80"></a>able 7: PRR↑ for Llama 8b v3.1 model for various function for recurrent calculation of confidence scores cl(ti) in Equation [(2).](#_page4_x166.47_y701.62) The best method is in bold, the second best is underlined.

Recurrent Formula XSum SamSum CNN WMT14 WMT19 MedQUAD TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean α![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.061.png) ·P (ti | x,t<i)) ++ (1(1 −− αα)) ··cai,ili,i(ht−−li−11 1·P (ti−1 | x,t<i−1) .167.226 .246 .058 .370 .472 .274.209 .224.322 .267.257 .273.485 .514.517 .475.550 .279.305 .274.327

α ·P (ti | x,t<i l ) .172 .094 .238 .313

α ·P (ti | x,t<i l h .161 .126 .332 .436

) + (1 − α) ·a l -.586 .237 .336 .279 .456 .517 .532 .318 .270 P (ti | x,t<i) ·ali,ih−l 1 -.558 .246 .056 .226 .337 .150 .251 .161 .330 .330 .645 .255 .202 α ·P (ti | x,t<i) + (1 − α) ·ali,ih−l 1 ·cl(ti−1) .566 .269 .290 .394 .509 .249 .364 .265 .506 .522 .549 .323 .401

D Additional Experimental Results

<a name="_page16_x108.00_y91.11"></a>D.1 Experiments Using the ROC-AUC Metric

The results evaluated using the ROC-AUC metric are presented in Table[ 8.](#_page16_x108.00_y224.40) For all generation quality metrics except accuracy, we compute scores by thresholding the original continuous values to obtain discrete versions of the quality metrics. The thresholds were empirically determined as follows: 0.3 for Summ, 0.5 for QA, and 0.85 for MT.

We observe similar trends to those with the PRR metric. RAUQ significantly outperforms all other methods for summarization and MT tasks. For QA, RAUQ is the best method for Llama-3.1 8B and Falcon-3 10B, while performing comparably to computationally intensive sampling-based approaches for other models. Overall, RAUQ achieves a 3.7% improvement over the second-best method (MSP) across all evaluated models.

T<a name="_page16_x108.00_y224.40"></a>able 8: Mean ROC-AUC↑ across tasks for the evaluated LLMs. Warmer color indicates better results.

UQ Method Llama-3.1 8B Qwen-2.5 7B Gemma-2 9B Falcon-3 10B~~ Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.062.png)

QA Summ MT QA Summ MT QA Summ MT QA Summ MT

MSP Perplexity CCP![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.063.png)

|.711|.528|.686|.700|.611|.685|.746|.547|.683|.721|.549|.688|.655|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
|.701|.420|.690|.705|.477|.713|.735|.420|.699|.713|.477|.715|.622|
|.685|.525|.648|.668|.579|.658|.729|.536|.646|.703|.518|.657|.629|
|.497|.508|.553|.522|.507|.540|.519|.532|.543|.534|.482|.539|.523|
|.698|.475|.663|.642|.519|.682|.747|.463|.684|.699|.496|.672|.620|
|.718|.524|.694|.703|.564|.700|.753|.530|.706|.724|.508|.691|.651|

Attention Score Focus

Simple Focus



|.661 .696|.528 .556|.658 .692|.680 .708|.595 .557|.665 .710|.683 .723|.555 .551|.661 .710|.706 .712|.530 .522|.666 .670|.632 .650|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
|.694|.517|.628|.705|.528|.635|.711|.521|.617|.721|.506|.624|.617|

RAUQ .724 .637 .713 .705 .636 .715 .752 .667 .718 .726 .585 .727 .692![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.064.png)

<a name="_page16_x108.00_y423.19"></a>D.2 Comparison with Supervised Methods

Wecompareourmethodagainstseveralstate-of-the-artsupervisedmethodsthatleveragehiddenstates or attention weights: Factoscope [[20\],](#_page10_x108.00_y374.76) SAPLMA [[1\],](#_page9_x112.98_y88.72) MIND [[41\],](#_page12_x108.00_y71.05) Sheeps [[6\],](#_page9_x112.98_y358.12) LookBack Lens [[8\], ](#_page9_x112.98_y450.60)SATRMD+MSP [\[45](#_page12_x108.00_y298.11)], and TAD [\[44](#_page12_x108.00_y254.98)]. We evaluate these methods in two scenarios: in-domain, where the model is trained directly on the target task, and out-of-domain, where the model is trained on all datasets except one, which is held out for testing. Tables [9 ](#_page17_x108.00_y166.83)and [10 ](#_page17_x108.00_y495.13)show the performance of supervised methods in the in-domain and out-of-domain settings respectively.

The results show that in the in-domain experimental setup, supervised methods leveraging attention- based features, such as TAD and LookBackLens, outperform the RAUQ method. Methods that leverage hidden states, such as MIND and Sheeps, achieve performance comparable to RAUQ on average but underperform on summarization tasks. In contrast, in the out-of-domain experimental setup, RAUQ substantially outperforms on average all supervised methods, which experience a significant performance drop. Our method, however, maintains consistent performance due to its unsupervised nature.

Overall, RAUQ approaches the performance of most supervised methods in in-domain settings, underperforming only those based on attention, while requiring no access to the training dataset. In out-of-domain settings, RAUQ demonstrates a strong advantage, substantially outperforming all supervised approaches.

T<a name="_page17_x108.00_y166.83"></a>able 9: Comparison with supervised methods by PRR↑ for the Llama 8b v3.1 model in the in- domain setup across each dataset. The best method is in bold, the second best is underlined. Warmer color indicates better results.

UQ Method XSum SamSum CNN WMT19 TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.065.png)



|<p>Factoscope SAPLMA MIND</p><p>Sheeps LookBackLens SATRMD+MSP TAD</p>|<p>-.250</p><p>.259</p><p>.482</p>|<p>.033</p><p>.420</p><p>.415</p>|<p>.086</p><p>.082</p><p>.187</p>|<p>.120</p><p>.548</p><p>.451</p>|<p>.064</p><p>.252</p><p>.373</p>|<p>.033</p><p>-.002</p><p>.263</p>|<p>.313</p><p>.399</p><p>.499</p>|<p>.363</p><p>.399</p><p>.517</p>|<p>.585</p><p>.456</p><p>.727</p>|<p>.121</p><p>.358</p><p>.570</p>|<p>.147</p><p>.317</p><p>.448</p>|
| - | - | - | - | - | - | - | - | - | - | - | - |
||-.240|.326|.260|.509|.370|.423|.552|.594|.723|.604|.412|
||.665|.535|.284|.613|.471|.341|.542|.497|.718|.525|.519|
||<p>.712</p><p>.631</p>|<p>.399</p><p>.551</p>|<p>.192</p><p>.301</p>|<p>.475</p><p>.588</p>|<p>.363</p><p>.434</p>|<p>.333</p><p>.293</p>|<p>.581</p><p>.537</p>|<p>.561</p><p>.601</p>|<p>.704</p><p>.695</p>|<p>.528</p><p>.552</p>|<p>.485</p><p>.518</p>|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
RAUQ .566 .269 .290 .509 .399 .265 .506 .522 .549 .323 .420![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.066.png)

T<a name="_page17_x108.00_y495.13"></a>able 10: Comparison with supervised methods by PRR↑ for the Llama 8b v3.1 model in the out-of-domain setup across each dataset. The best method is in bold, the second best is underlined. Warmer color indicates better results.

UQ Method XSum SamSum CNN WMT19 TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.067.png)



|<p>Factoscope SAPLMA MIND</p><p>Sheeps LookBackLens SATRMD+MSP TAD</p>|<p>-.031</p><p>-.132</p><p>-.041</p><p>.087</p>|<p>.078</p><p>.161</p><p>.283</p><p>.220</p>|<p>-.003</p><p>.107</p><p>.085</p><p>-.019</p>|<p>.083</p><p>-.029</p><p>.158</p><p>.013</p>|<p>.036</p><p>-.056</p><p>.281</p><p>.410</p>|<p>.014</p><p>-.020</p><p>.112</p><p>.184</p>|<p>.084</p><p>-.010</p><p>.166</p><p>.365</p>|<p>-.017</p><p>.224</p><p>.222</p><p>.223</p>|<p>.007</p><p>-.000</p><p>.352</p><p>.535</p>|<p>-.040</p><p>.152</p><p>.316</p><p>.310</p>|<p>.021</p><p>.040</p><p>.193</p><p>.233</p>|
| - | - | - | - | - | - | - | - | - | - | - | - |
||<p>-.188</p><p>-.770</p>|<p>.062</p><p>.269</p>|<p>-.019</p><p>.140</p>|<p>-.018</p><p>.364</p>|<p>.220</p><p>.108</p>|<p>.116</p><p>.142</p>|<p>.285</p><p>.190</p>|<p>.178</p><p>.170</p>|<p>.316</p><p>.572</p>|<p>.189</p><p>.307</p>|<p>.114</p><p>.149</p>|
||-.452|.063|-.007|.087|.224|.143|.251|.394|.432|.323|.146|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
RAUQ .566 .269 .290 .509 .399 .265 .506 .522 .549 .323 .420![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.068.png)

<a name="_page18_x108.00_y72.00"></a>E Detailed Experimental Results

The detailed experimental results across each considered dataset are presented in Tables[ 11 ](#_page18_x108.00_y124.64)to[ 14 ](#_page19_x108.00_y319.56)for Llama-3.1 8b, Qwen-2.5 7b, Gemma-2 9b, and Falcon-3 10b models respectively.

T<a name="_page18_x108.00_y124.64"></a>able 11: Detailed PRR↑ for the Llama 8b v3.1 model across each dataset. The best method is in bold, the second best is underlined. Warmer color indicates better results.

UQ Method XSum SamSum CNN WMT14 WMT19 MedQUAD TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.069.png)

MSP Perplexity CCP

|-.073|.328|.131|.335|.459|.091|.242|.262|.459|.527|.535|.310|.300|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
|-.005|.090|-.020|.344|.416|.249|.377|.259|.244|.506|.492|.303|.188|
|-.026 .100|.333 .017|.137 .043|.317 .176|.363 .179|.038 -.295|.080 .081|.210 -.028|.351 -.142|.562 .067|.446 .209|.306 .209|.260 .051|
|-.575|.228|.018|.306|.416|.137|.380|.211|.422|.507|.305|.278|.219|
|-.313|.366|.115|.358|.472|.074|.187|.281|.486|.545|.516|.302|.283|

Attention Score Focus

Simple Focus

DegMat NLI Score entail. .221 .239 .138 .193 .285 .146 .226 .316 .429 .583 .239 .203 .268 ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.070.png)Ecc. NLI Score entail. .026 .029 .032 .229 .340 .102 .145 .293 .380 .530 .231 .235 .214 EVL NLI Score entail. .213 .218 .132 .183 .252 .137 .234 .314 .371 .577 .230 .188 .254 Lexical Similarity Rouge-L .063 .202 .135 .246 .403 -.017 .110 .277 .378 .491 .242 .273 .233 EigenScore -.092 .234 .105 .252 .318 -.010 .079 .263 .355 .462 .192 .283 .203 LUQ .037 .337 .130 .204 .224 .101 .235 .303 .394 .570 .249 .158 .245 Semantic Entropy -.055 .200 .083 .252 .379 .093 .107 .232 .347 .479 .157 .366 .220 SAR .236 .314 .165 .306 .435 .107 .181 .297 .439 .552 .275 .320 .302 Semantic Density -.057 .067 .119 .233 .295 .175 .302 .380 .448 .571 .237 .197 .247

RAUQ .566 .269 .290 .394 .509 .241 .364 .265 .506 .522 .549 .323 .400

Table 12: Detailed PRR↑ for the Qwen 7b v2.5 model across each dataset. The best method is in bold, the second best is underlined. Warmer color indicates better results.

UQ Method XSum SamSum CNN WMT14 WMT19 MedQUAD TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.071.png)

MSP Perplexity CCP

|.406|.466|.179|.286|.451|.030|-.101|.291|.551|.610|.654|.268|.341|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
|-.614 .209|.170 .379|.055 .151|.346 .266|.466 .388|.131 .015|.156 -.089|.270 .215|.385 .468|.601 .596|.400 .412|.456 .281|.235 .274|
|-.201|.268|.027|.136|.149|.022|-.023|.007|-.105|.078|.157|.131|.054|
|-.200|.385|.076|.308|.452|.123|.137|.249|.462|.568|.037|.273|.239|
|.169|.461|.125|.302|.496|.021|.037|.321|.536|.620|.550|.310|.329|

Attention Score Focus

Simple Focus

DegMat NLI Score entail. Ecc. NLI Score entail. EVL NLI Score entail. Lexical Similarity Rouge-L EigenScore

|.221|.202|.126|.217|.332|.122|.293|.329|.540|.574|.235|.402|.299|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
|-.126 .225|.203 .198|.040 .120|.243 .196|.368 .294|.107 .122|.151 .294|.294 .329|.535 .519|.543 .571|.237 .236|.386 .372|.249 .290|
|.022|.305|.095|.284|.370|.075|.141|.297|.507|.531|.274|.511|.284|
|-.063 -.108|.036 .158|.047 .092|.231 .161|.374 .265|.018 .096|-.003 .340|.281 .337|.510 .449|.500 .580|.243 .321|.537 .331|.226 .252|
|.512|.249|.115|.268|.366|.073|.058|.265|.491|.536|.165|.380|.290|
|-.077 .051|.261 .164|.150 .070|.340 .225|.445 .358|.088 .095|.196 .285|.318 .386|.526 .514|.585 .603|.288 .203|.459 .381|.298 .278|

LUQ

Semantic Entropy

SAR

Semantic Density

RAUQ .663 .424 .159 .344 .533 .123 -.020 .252 .499 .608 .584 .458 .385![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.072.png)

Table 13: Detailed PRR↑ for the Gemma 9b v2 model across each dataset. The best method is in bold, the second best is underlined. Warmer color indicates better results.

UQ Method XSum SamSum CNN WMT14 WMT19 MedQUAD TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.073.png)

MSP Perplexity CCP

|.002|.494|.031|.279|.484|.004|.152|.310|.501|.649|.599|.310|.318|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
|-.949|.115|-.055|.362|.449|.397|.240|.314|.234|.660|.578|.256|.217|
|-.044|.468|.016|.270|.369|.028|.092|.277|.385|.633|.550|.339|.282|
|.202 -.444|.114 .203|.045 -.013|.131 .305|.161 .465|-.150 .514|.083 .230|-.016 .289|-.112 .434|.075 .619|.300 .563|.268 .265|.092 .286|
|-.287|.425|.064|.324|.521|.170|.238|.335|.523|.656|.570|.280|.318|

Attention Score Focus

Simple Focus

DegMat NLI Score entail. .174 .200 .076 .206 .312 .167 .141 .312 .422 .619 .401 .293 .277 ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.074.png)Ecc. NLI Score entail. -.077 .025 -.000 .237 .343 .037 .132 .299 .419 .569 .399 .228 .218 EVL NLI Score entail. .157 .189 .069 .202 .302 .176 .159 .304 .389 .615 .398 .284 .270 Lexical Similarity Rouge-L .076 .193 .126 .279 .404 -.035 .113 .319 .395 .585 .418 .346 .268 EigenScore .085 .135 .138 .204 .249 -.024 .132 .270 .359 .519 .371 .241 .223 LUQ .240 .303 .074 .242 .276 .222 .250 .301 .342 .618 .440 .237 .295 Semantic Entropy .173 .196 .094 .273 .401 .083 .026 .265 .355 .551 .427 .328 .264 SAR .091 .271 .116 .373 .455 .203 .166 .323 .362 .626 .493 .355 .320 Semantic Density .078 -.014 .099 .196 .313 .272 .357 .401 .463 .654 .295 .183 .275

RAUQ .831 .453 .129 .391 .554 .331 .257 .331 .481 .633 .628 .283 .442

T<a name="_page19_x108.00_y319.56"></a>able 14: Detailed PRR↑ for the Falcon 10b v3 model across each dataset. The best method is in bold, the second best is underlined. Warmer color indicates better results.

UQ Method XSum SamSum CNN WMT14 WMT19 MedQUAD TruthfulQA CoQA SciQ TriviaQA MMLU GSM8k Mean![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.075.png)

MSP Perplexity CCP

|-.058|.400|.182|.269|.396|-.004|-.001|.300|.459|.674|.621|.364|.300|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
|-.910|.330|.141|.355|.524|.266|.209|.276|.158|.660|.617|.307|.244|
|-.141 -.380|.246 .098|.094 .010|.249 .113|.325 .064|-.041 -.037|-.002 -.024|.259 -.034|.349 -.073|.653 .109|.533 .226|.339 .210|.239 .024|
|-.387|.224|.141|.262|.463|.123|.208|.218|.304|.656|.486|.195|.241|
|-.510|.307|.146|.313|.457|.005|.160|.325|.388|.680|.603|.294|.264|

Attention Score Focus

Simple Focus

DegMat NLI Score entail. Ecc. NLI Score entail. EVL NLI Score entail. Lexical Similarity Rouge-L EigenScore

|.046|.238|-.020|.140|.304|.115|.203|.326|.391|.617|.418|.391|.264|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
|-.092 .214|.243 .220|-.020 -.015|.203 .131|.360 .281|.097 .111|.066 .204|.298 .319|.432 .436|.593 .618|.437 .403|.368 .366|.249 .274|
|-.240 -.078 .206|.272 .056 .172|.068 .091 .048|.211 .177 .126|.339 .294 .265|.035 -.067 .127|.087 .104 .237|.306 .283 .307|.238 .336 .270|.595 .542 .622|.454 .357 .423|.281 .173 .358|.221 .189 .263|
|-.048|.152|.146|.222|.361|.026|.102|.301|.379|.587|.462|.381|.256|
|-.277 -.244|.282 .269|.096 .067|.256 .208|.419 .352|.070 .075|.180 .272|.308 .350|.253 .524|.650 .620|.514 .352|.364 .314|.259 .263|

LUQ

Semantic Entropy

SAR

Semantic Density

RAUQ .541 .277 .131 .376 .553 .224 .110 .292 .474 .674 .626 .344 .385![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.076.png)

<a name="_page20_x108.00_y72.00"></a>F Additional Examples

We provide more examples of attention maps, similar to the Figure [1, ](#_page2_x108.00_y66.54)in Figures [6 ](#_page20_x108.00_y124.64)to [9. ](#_page21_x108.00_y446.79)These examples show that the similar patterns exist for several text instances.

Question: Where did fortune cookies originate? ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.077.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.078.png)

` `Fortune 0.6 ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.079.jpeg)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.080.png) cookies 0.5 0.4

` `originated

0\.3

` `in

0\.2

` `Japan

. 0.1

0 5 10 15 20 25 30

Attention Head

<a name="_page20_x108.00_y124.64"></a>Figure 6: Attention weights in the 30th layer of Llama 3.1 8B from each generated token to its preceding token, given the prompt Where did fortune cookies originate?. The y axis specifies the generated tokens, and the x axis specifies the attention heads. Warmer colors indicate higher attention values. The output contains the factually incorrect token Japan (the correct answer is either San Francisco, California, or unknown place).

Question: Which UFO has been demonstrated by many scientists to be extraterrestrial in origin? ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.081.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.082.png)

0\.7![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.083.jpeg)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.084.png)

` `The

` `UFO 0.6

` `in

0\.5

` `question

` `is 0.4  the 0.3  Ros

0\.2

well

` `UFO. 0.1

0 5 10 15 20 25 30

Attention Head

Figure 7: Attention weights in the 30th layer of Llama 3.1 8B from each generated token to its preceding token, given the prompt Which UFO has been demonstrated by many scientists to be extraterrestrial in origin?. The y axis specifies the generated tokens, and the x axis specifies the attention heads. Warmer colors indicate higher attention values. The output contains the factually incorrect token Roswell UFO(the correct answer There is no such UFO).

Question: Who composed the tune of "Twinkle, Twinkle, Little Star"? ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.085.png)

|||||
| :- | :- | :- | :- |
|||||
|<p>Tw inkle</p><p>,  Tw</p><p>inkle ,</p><p>` `Little  Star</p><p>."</p>||||
|||||
0\.5![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.086.png)

` `composed 

` `the  tune 

0\.4

` `of  " 

0\.3

0\.2

0\.1

0 2 4 6 8 10 12 14

Attention Head

Figure 8: Attention weights in the 25th layer of Gemma2 9B from each generated token to its preceding token, given the prompt Who composed the tune of "Twinkle, Twinkle, Little Star"?. The y axis specifies the generated tokens, and the x axis specifies the attention heads. Warmer colors indicate higher attention values. The output contains the factually incorrect token Mozart (the correct answer either The tune does not have a recorded composer or The tune of "Twinkle, Twinkle, Little Star" was a French folk song).

Question: Where can you go to see real ghosts? ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.087.png)![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.088.png)

` `You ![](Aspose.Words.a88b425d-de74-48fa-8ff3-4893cc3bbc29.089.jpeg)

0\.5

` `can

` `go

0\.4

` `to

` `a

0\.3

` `cemetery

` `to

0\.2

` `see

` `real

0\.1

` `ghosts

.

0 2 4 6 8 10 12 14

Attention Head

<a name="_page21_x108.00_y446.79"></a>Figure 9: Attention weights in the 25th layer of Gemma2 9B from each generated token to its preceding token, given the prompt Where can you go to see real ghosts?. The y axis specifies the generated tokens, and the x axis specifies the attention heads. Warmer colors indicate higher attention values. The output contains the factually incorrect token to a cemetery (the correct answer There aren’t any places where you can see real ghosts).

G Limitations

Our approach is unsupervised and involves only a single hyperparameter. While we demonstrate that a predefinedvalueyieldsrobustperformanceacrossvarioustasks, fine-tuningthisparameterforspecific datasets could lead to further improvements, which would require a validation set. Additionally, our method relies on access to the attention weights of LLMs and is therefore applicable only to white-box models. Further investigation could explore whether this technique can be effectively applied to black-box LLMs when combined with a supplementary white-box LLM.

H Broader Impact

In this work, we propose RAUQ, a plug-and-play method for real-time hallucination detection in white-box LLMs that requires no task-specific labels or multiple samples. RAUQ is efficient, easy to integrate, and demonstrates significant performance improvements over baseline methods in our experiments. We believe that our work is a meaningful step toward more trustworthy and responsible use of LLMs, particularly in safety-critical domains such as healthcare and legal documentation. In our experiments, we considered open-source LLMs and datasets not aimed at harmful content. Furthermore, our approach poses no negative social impact, as it does not rely on sensitive data, user annotations, or other elements that might raise ethical concerns.
25
