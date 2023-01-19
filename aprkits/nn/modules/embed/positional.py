import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


class PositionalEmbedding(Module):
    def __init__(
            self,
            d_model: int,
            max_length: int,
            device=None,
            dtype=None
    ):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        pos_encoding = torch.zeros(max_length, d_model, dtype=dtype, device=device)
        positions_list = torch.arange(0, max_length, dtype=dtype, device=device).view(-1, 1)
        division_term = torch.exp(torch.arange(
            0, d_model, 2, dtype=dtype, device=device
        ) * (-np.log(10_000.0)) / d_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, embedding: Tensor):
        return embedding + self.pos_encoding[:embedding.shape[0], :]
