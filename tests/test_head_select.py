import torch

from src.uq.head_select import select_heads


def test_head_selection_uses_generated_tokens_only():
    prompt_len = 2
    total_len = prompt_len + 3
    layer = torch.zeros(1, 2, total_len, total_len)
    # head 0 stares at prompt token, head 1 focuses on previous generated token
    layer[0, 0, prompt_len, prompt_len - 1] = 0.9
    layer[0, 0, prompt_len + 1, prompt_len - 1] = 0.95
    layer[0, 1, prompt_len, prompt_len - 1] = 0.5
    layer[0, 1, prompt_len + 1, prompt_len] = 0.99
    layer[0, 1, prompt_len + 2, prompt_len + 1] = 0.98
    heads = select_heads([layer], prompt_len)
    assert heads == [1]
