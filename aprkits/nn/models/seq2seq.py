from typing import Optional, List

import torch
from torch import LongTensor
from torch.nn import Module

import aprkits.nn.functional as f


class Seq2SeqModel(Module):
    def __init__(
            self,
            encoder: Module,
            decoder: Module,
            output_hidden_states: bool = None,
            output_attentions: bool = None
    ):
        super().__init__()

        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.encoder_config = encoder.config
        self.decoder_config = decoder.config

        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            input_ids: LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ):
        embed = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds,
            use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        embed = embed['last_hidden_state']

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = f.shift_right(
                labels,
                decoder_start_token_id=self.decoder_config.decoder_start_token_id,
                pad_token_id=self.decoder_config.pad_token_id
            )

        output = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, head_mask=decoder_head_mask,
            inputs_embeds=decoder_inputs_embeds, encoder_hidden_states=embed, encoder_attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions or self.output_attentions,
            output_hidden_states=output_hidden_states or self.output_hidden_states, return_dict=return_dict)

        return output


class Seq2CMDSeqModel(Module):
    def __init__(
            self,
            encoder: Module,
            decoder: Module,
            regressor: Module,
    ):
        super().__init__()

        self.encoder_config = encoder.config
        self.decoder_config = decoder.config
        self.regressor_config = regressor.config

        self.encoder = encoder
        self.decoder = decoder
        self.regressor = regressor

    def forward(
            self,
            input_ids: LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ):
        embed = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds,
            use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        embed = embed['last_hidden_state']

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = f.shift_right(
                labels,
                decoder_start_token_id=self.decoder_config.decoder_start_token_id,
                pad_token_id=self.decoder_config.pad_token_id
            )

        seq_output = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, head_mask=decoder_head_mask,
            inputs_embeds=decoder_inputs_embeds, encoder_hidden_states=embed, encoder_attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict)

        regressor_input_ids = torch.concat((input_ids, labels), dim=-1)
        regressor_attention_mask = torch.concat((attention_mask, decoder_attention_mask), dim=-1)
        loc_output = self.regressor(input_ids=regressor_input_ids, attention_mask=regressor_attention_mask)

        loc_output['logits'] = loc_output['logits'].squeeze(-1)
        return {
            'lm': seq_output,
            'rg': loc_output
        }
