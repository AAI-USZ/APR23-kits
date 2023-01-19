from dataclasses import dataclass

import torch

from aprkits.annotations import dictclass


@dataclass
@dictclass
class ForwardBatchLMOutput:
    logits: torch.FloatTensor
    loss: torch.FloatTensor
    labels: torch.LongTensor
    class_first: bool = False
