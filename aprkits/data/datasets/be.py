from typing import Tuple, Union, List

import numpy as np
import torch
from torch.utils.data import Dataset


class BatchEncodingDataset(Dataset):
    def __init__(self, inputs, targets):
        assert len(inputs['input_ids']) == len(targets['input_ids'])
        assert len(inputs['attention_mask']) == len(targets['attention_mask'])
        self.inp_data = inputs['input_ids']
        self.tar_data = targets['input_ids']
        self.inp_data_mask = inputs['attention_mask']
        self.tar_data_mask = targets['attention_mask']

    def __getitem__(self, index):
        return (
            self.inp_data[index], self.tar_data[index],
            self.inp_data_mask[index], self.tar_data_mask[index]
        )

    def __len__(self):
        return len(self.inp_data)


class InputTargetDataset(Dataset):
    def __init__(
            self,
            inputs: Union[
                np.ndarray,
                torch.Tensor,
                List[Union[np.ndarray, torch.Tensor]],
                Tuple[Union[np.ndarray, torch.Tensor], ...]
            ],
            targets: Union[
                np.ndarray,
                torch.Tensor,
                List[Union[np.ndarray, torch.Tensor]],
                Tuple[Union[np.ndarray, torch.Tensor], ...]
            ]):
        self._is_src_single = self._is_tgt_single = False
        if isinstance(inputs, (np.ndarray, torch.Tensor)):
            self._is_src_single = True
            inputs = (inputs,)
        if isinstance(targets, (np.ndarray, torch.Tensor)):
            self._is_tgt_single = True
            targets = (targets,)

        assert all(len(src) == len(tgt) for src in inputs for tgt in targets), 'Size mismatch between tensors.'

        self.src = inputs
        self.tgt = targets

    def __getitem__(self, item):
        if self._is_src_single:
            inputs = self.src[0][item]
        else:
            inputs = tuple(src[item] for src in self.src)
        if self._is_tgt_single:
            targets = self.tgt[0][item]
        else:
            targets = tuple(tgt[item] for tgt in self.tgt)
        return inputs, targets

    def __len__(self):
        return len(self.src[0]) if len(self.src) > 0 else 0
