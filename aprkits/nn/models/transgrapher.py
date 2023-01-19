import torch
from torch.nn import Module, Dropout

from aprkits.nn.modules import NumeralEmbedding


class Transgrapher(Module):
    def __init__(
            self,
            token_encoder: Module,
            graph_encoder: Module,
            decoder: Module,
            dropout_rate: float = 0.1,
            pad_token_id: int = 1,
            bos_token_id: int = 0,
            sep_token_id: int = 2,
            max_num: int = 32
    ):
        super().__init__()

        self._pad_token_id = pad_token_id
        self._bos_token_id = bos_token_id
        self._sep_token_id = sep_token_id
        self._max_num = max_num

        self.token_encoder = token_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder

        config = graph_encoder.config
        self.num_embed = NumeralEmbedding(d_model=config.hidden_size, max_num=self._max_num)

        self.dropout = Dropout(dropout_rate)

    def forward(
            self,
            input_ids,
            labels,
            graph_ids=None,
            graph_type_ids=None,
            count_ids=None,
            attention_mask=None,
            graph_attention_mask=None,
            tgt_mask=None
    ):
        token_encoder_output = self.token_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        graph_encoder_output = self.graph_encoder(
            input_ids=graph_ids, token_type_ids=graph_type_ids, attention_mask=graph_attention_mask)

        token_embed = token_encoder_output.last_hidden_state
        graph_embed = graph_encoder_output.last_hidden_state

        token_embed = self.dropout(token_embed)
        graph_embed = self.num_embed(count_ids, graph_embed)
        graph_embed = self.dropout(graph_embed)

        embed = torch.concat((token_embed, graph_embed), dim=1)
        decoder_encoder_attention_mask = _concat_masks(attention_mask, graph_attention_mask)

        seq2seq_lm_output = self.decoder(
            inputs_embeds=embed,
            labels=labels,
            attention_mask=decoder_encoder_attention_mask,
            decoder_attention_mask=tgt_mask
        )

        return seq2seq_lm_output


def _concat_masks(*masks):
    masks = tuple(mask for mask in masks if mask is not None)
    mask = torch.concat(masks, dim=-1)
    return mask
