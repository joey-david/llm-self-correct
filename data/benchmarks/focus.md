### TruthfulQA (generation / free-form)

```python
from datasets import load_dataset
ds = load_dataset("truthful_qa", "generation", split="validation")
# fields: question, best_answer, correct_answers, incorrect_answers, category, ...
```

Source shows `truthful_qa` with a `"generation"` config and a `validation` split. ([GitHub][1])

*(FYI MC version if ever needed:)*

```python
mc = load_dataset("EleutherAI/truthful_qa_mc", split="validation")
```

([Hugging Face][2])

---

### PopQA (long-tail, closed-book)

```python
from datasets import load_dataset
ds = load_dataset("akariasai/PopQA", split="train")      # dataset ships as a single split
# optional: make a small val slice
val = ds.train_test_split(test_size=0.1, seed=42)["test"]
```

Dataset card + fields on HF. Many mirrors exist; the canonical one most folks use is `akariasai/PopQA`. ([Hugging Face][3])

---

### HotpotQA (multi-hop; “distractor” setting)

```python
from datasets import load_dataset
dev = load_dataset("hotpot_qa", "distractor", split="validation")
train = load_dataset("hotpot_qa", "distractor", split="train")
```

“distractor” is a closed-book-ish config; HF viewer confirms the `validation` split for that config. ([Hugging Face][4])

---

### SQuAD 2.0 (unanswerable subset for abstention behavior)

```python
from datasets import load_dataset
train = load_dataset("squad_v2", split="train")
dev   = load_dataset("squad_v2", split="validation")
```

Official HF dataset entry (and docs/examples) use `squad_v2`. ([Hugging Face Forums][5])

---

### AlignScore (for soft correctness / tie-breaks)

* Models/checkpoints: `yzha/AlignScore` repo on HF. ([Hugging Face][6])
* Paper: ACL 2023. ([aclanthology.org][7])


[1]: https://github.com/vllm-project/vllm/issues/4606?utm_source=chatgpt.com "[Bug]: when dtype='bfloat16', batch_size will cause different ..."
[2]: https://huggingface.co/datasets/EleutherAI/truthful_qa_mc?utm_source=chatgpt.com "EleutherAI/truthful_qa_mc · Datasets at ..."
[3]: https://huggingface.co/datasets/akariasai/PopQA "akariasai/PopQA · Datasets at Hugging Face"
[4]: https://huggingface.co/datasets/hotpotqa/hotpot_qa?utm_source=chatgpt.com "hotpotqa/hotpot_qa · Datasets at Hugging Face"
[5]: https://discuss.huggingface.co/t/transformed-dataset-to-json-saves-cache-dataset/28808?utm_source=chatgpt.com "Transformed dataset to_json saves cache dataset - Beginners"
[6]: https://huggingface.co/yzha/AlignScore?utm_source=chatgpt.com "yzha/AlignScore · Hugging Face"
[7]: https://aclanthology.org/2023.acl-long.634.pdf?utm_source=chatgpt.com "ALIGNSCORE: Evaluating Factual Consistency with A ..."
