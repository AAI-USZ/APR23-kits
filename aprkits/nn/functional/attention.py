import numpy as np
import torch


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = torch.matmul(q, torch.transpose(k, -2, -1))
    attn_logits = attn_logits / np.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 1, -torch.inf)
    attn = torch.softmax(attn_logits, dim=-1)
    values = torch.matmul(attn, v)
    return values, attn
