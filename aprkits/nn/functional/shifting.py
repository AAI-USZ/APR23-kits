import torch
from transformers.utils import is_torch_fx_proxy


def shift_right(input_ids, decoder_start_token_id=0, pad_token_id=1):
    assert decoder_start_token_id is not None, (
        'Decoder_start_token_id has to be defined. It is usually set to the pad_token_id.'
    )
    if is_torch_fx_proxy(input_ids):
        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
    else:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
    assert pad_token_id is not None, 'Pad_token_id has to be defined.'
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    assert torch.all(shifted_input_ids >= 0).item(), 'Verify that `shifted_input_ids` has only positive values'
    return shifted_input_ids
