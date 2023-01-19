import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


class NumeralEmbedding(Module):
    def __init__(
            self,
            d_model: int,
            max_num: int,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.max_num = max_num

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        mod_term = ((1 - torch.cos(torch.arange(d_model, dtype=dtype, device=device))) / 2) / np.sqrt(d_model)
        mod_term = mod_term[None, ...]

        self.register_buffer('mod_term', mod_term, persistent=False)

    def forward(self, numerics: Tensor, embedding: Tensor):
        numerics = numerics[..., None]
        num_encoding = (numerics / self.max_num) @ self.mod_term
        return embedding + num_encoding
