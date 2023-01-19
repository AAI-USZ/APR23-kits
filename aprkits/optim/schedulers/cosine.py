import numpy as np
from torch.optim import Optimizer
# noinspection PyProtectedMember
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_iters: int,
            total_iters: int,
            last_epoch: int = -1,
            verbose: bool = False
    ):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        lr_factor = self.get_lr_factor(self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.total_iters))
        if epoch <= self.warmup_iters:
            lr_factor *= epoch * 1.0 / self.warmup_iters
        return lr_factor
