import torch
from torch import Tensor


def pad_mask(x: Tensor, pad_idx: int, device=None, dtype=None) -> Tensor:
    x = (x == pad_idx).to(torch.int32)
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.to(dtype)
    return x


def lookahead_mask(size: int, device=None, dtype=None) -> Tensor:
    x = torch.triu(torch.full((size, size), fill_value=1, dtype=dtype, device=device), 1)
    return x


def combine_masks(*masks: Tensor) -> Tensor:
    mask = torch.concat(masks).view(len(masks), *masks[0].shape)
    mask = mask.sum(dim=0)
    mask = (mask > 0).to(torch.int32)
    return mask
