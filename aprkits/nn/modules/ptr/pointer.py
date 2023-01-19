from typing import Union, Callable

import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.nn import Module, Linear


def masked_log_softmax(
        x: torch.Tensor,
        mask: torch.Tensor = None,
        dim: int = -1,
        eps: float = 1e-8
):
    if mask is not None:
        x = x + (mask.float() + eps).log()
    return F.log_softmax(x, dim=dim)


class PtrAttention(Module):
    def __init__(
            self,
            d_model: int,
            src_length: int,
            tgt_length: int,
            bias: bool = False,
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = 'tanh'
    ):
        super().__init__()
        self.src_length = src_length
        self.tgt_length = tgt_length
        self.w1 = Linear(d_model, src_length, bias=bias)
        self.w2 = Linear(d_model, src_length, bias=bias)
        self.vt = Linear(src_length, 1, bias=bias)
        self.act = activation
        if isinstance(self.act, str):
            try:
                self.act = getattr(torch, self.act)
            except AttributeError:
                self.act = getattr(F, self.act)

    def forward(
            self, decoder_hidden_state: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None,
            eps: float = None):
        decoder_transform = self.w1(decoder_hidden_state).unsqueeze(2)
        encoder_transform = self.w2(encoder_outputs).unsqueeze(1)
        weights = (decoder_transform + encoder_transform)
        weights = self.vt(self.act(weights)).squeeze(-1)
        log_score = masked_log_softmax(weights, mask, dim=-1, eps=eps)
        return log_score


class PtrNetwork(Module):
    def __init__(
            self,
            d_model: int,
            src_length: int,
            tgt_length: int,
            bias: bool = False,
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = 'tanh',
            pad_token_id: int = None
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.attn = PtrAttention(
            d_model=d_model, src_length=src_length, tgt_length=tgt_length, bias=bias, activation=activation)

    def forward(
            self,
            decoder_hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            input_ids: torch.Tensor = None,
            mask: torch.Tensor = None,
            input_lengths: torch.Tensor = None,
            eps: float = 1e-8
    ):
        assert input_lengths is not None or mask is not None or input_ids is not None, \
            'Either input lengths, input_ids or mask should be specified.'

        batch_size = encoder_hidden_states.size(0)
        src_max_seq_len = encoder_hidden_states.size(1)

        if input_lengths is None and mask is None and input_ids is not None:
            assert self.pad_token_id is not None, \
                'If only input ids are provided, then pad token should be provided too.'
            input_lengths = src_max_seq_len - (input_ids == self.pad_token_id).sum(-1)

        # (batch_size, tgt_length, src_length)
        if mask is None:
            range_tensor = torch.arange(src_max_seq_len, device=input_lengths.device, dtype=input_lengths.dtype)
            range_tensor = range_tensor.expand(batch_size, src_max_seq_len)
            row_mask_tensor = torch.less(range_tensor, input_lengths.unsqueeze(-1))
            row_mask_tensor = row_mask_tensor.unsqueeze(1)
            mask = row_mask_tensor

        log_pointer_score = self.attn(decoder_hidden_states, encoder_hidden_states, mask, eps)
        log_pointer_score = log_pointer_score.transpose(-1, -2)
        return log_pointer_score
