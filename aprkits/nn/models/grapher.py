import torch
from torch import nn


class CatGraphNet(nn.Module):
    def __init__(
            self,
            graph_encoder,
            token_encoder,
            child_embed,
            token_decoder
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.token_encoder = token_encoder
        self.child_embed = child_embed
        self.token_decoder = token_decoder

    def forward(
            self,
            token_input_ids,
            node_input_ids,
            type_input_ids,
            count_input_ids,
            token_target_ids=None,
            token_input_attention_mask=None,
            node_attention_mask=None,
            token_target_attention_mask=None,
            *args,
            **kwargs
    ):
        graph_embed = self.graph_encoder(
            input_ids=node_input_ids,
            attention_mask=node_attention_mask,
            token_type_ids=type_input_ids
        )
        token_embed = self.token_encoder(
            input_ids=token_input_ids,
            attention_mask=token_input_attention_mask,
        )
        child_embed = self.child_embed(count_input_ids)
        graph_embed = graph_embed.last_hidden_state + child_embed
        token_embed = token_embed.last_hidden_state

        embed = torch.concat((graph_embed, token_embed), dim=1)
        mask = torch.concat((node_attention_mask, token_input_attention_mask), dim=1)

        lm_out = self.token_decoder(
            inputs_embeds=embed,
            attention_mask=mask
        )

        return lm_out
