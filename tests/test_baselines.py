import pytest
import torch

from src.uq.baselines import attention_score, msp_perplexity


def make_layer(prompt_len, values):
    total_len = prompt_len + len(values)
    attn = torch.zeros(1, 1, total_len, total_len)
    for idx, val in enumerate(values):
        row = prompt_len + idx
        attn[0, 0, row, row - 1] = val
    return attn


def test_msp_returns_negative_log_prob_average():
    probs = torch.tensor([0.5, 0.25])
    res = msp_perplexity(probs)
    expected = float(torch.mean(-torch.log(probs)).item())
    assert res.sequence == pytest.approx(expected, rel=1e-6)
    assert len(res.token_scores) == 2


def test_attention_score_selected_head_metadata():
    prompt_len = 1
    attentions = [make_layer(prompt_len, [0.0, 0.9])]
    res = attention_score(attentions, prompt_len)
    assert res.extras["selected_heads"] == [0]
    assert len(res.token_scores) == 2
