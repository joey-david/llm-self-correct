import math

import torch

from src.uq.rauq import rauq_score


def make_attn(values_head0, values_head1, prompt_len):
    gen_len = len(values_head0)
    total_len = prompt_len + gen_len
    heads = []
    for values in (values_head0, values_head1):
        mat = torch.zeros(total_len, total_len)
        for i, att in enumerate(values):
            row = prompt_len + i
            prev = row - 1
            mat[row, prev] = att
        heads.append(mat)
    return torch.stack(heads).unsqueeze(0)


def _manual(alpha, probs, att_values):
    conf = [probs[0]]
    for idx in range(1, len(probs)):
        prev = conf[-1]
        conf.append(alpha * probs[idx] + (1 - alpha) * att_values[idx] * prev)
    return conf


def test_rauq_sequence_score_matches_manual():
    prompt_len = 1
    probs = torch.tensor([0.8, 0.7, 0.6])
    layer0 = make_attn([0.0, 0.9, 0.8], [0.0, 0.2, 0.1], prompt_len)
    layer1 = make_attn([0.0, 0.3, 0.4], [0.0, 0.85, 0.75], prompt_len)
    out = rauq_score(probs, [layer0, layer1], prompt_len, alpha=0.5)
    assert out.selected_heads == [0, 1]
    conf_layer0 = _manual(0.5, probs.tolist(), [0.0, 0.9, 0.8])
    conf_layer1 = _manual(0.5, probs.tolist(), [0.0, 0.85, 0.75])
    layer0_u = -sum(math.log(x) for x in conf_layer0) / len(conf_layer0)
    layer1_u = -sum(math.log(x) for x in conf_layer1) / len(conf_layer1)
    assert math.isclose(out.u, max(layer0_u, layer1_u), rel_tol=1e-5)
    assert len(out.token_spikes) == len(probs)
