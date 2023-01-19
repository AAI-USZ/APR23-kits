from typing import Iterable, Collection, Dict, Union, Any

import numpy as np


def get_longer_seq_indices(data: Iterable[Collection[int]], max_len: int):
    indices = np.where(np.array([len(arr) for arr in data]) <= max_len)[0]
    return indices


def get_longer_seq_mask(data: Iterable[Collection[int]], max_len: int):
    mask = np.array([len(arr) for arr in data]) <= max_len
    return mask


def mask_data(
        data: Union[Dict[str, Collection[Collection[int]]], Collection[Any]],
        mask: Collection[Union[int, bool]],
        affected_keys: Iterable[str] = None
):
    if isinstance(data, dict):
        if affected_keys is None:
            affected_keys = {'input_ids', 'attention_mask', 'token_type_ids'}
        keys = affected_keys.copy()
        match_keys = keys.intersection(data.keys())
        return {
            key: np.array(val, dtype=object)[np.array(mask, dtype=np.bool)].tolist()
            if key in match_keys
            else val
            for key, val in data.items()
        }
    return np.array(data, dtype=object)[np.array(mask, dtype=np.bool)].tolist()
