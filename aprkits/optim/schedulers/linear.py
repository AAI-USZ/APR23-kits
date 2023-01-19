from torch.optim import Optimizer
# noinspection PyProtectedMember
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            total_iters: int,
            warmup_iters: int = 0,
            start_factor: float = 0.0,
            warmup_factor: float = 1.0,
            end_factor: float = 0.5,
            last_epoch: int = -1,
            verbose: bool = False
    ):
        self.warmup_iters = warmup_iters
        self.total_iters_ = total_iters
        self.total_iters = max(total_iters - warmup_iters, 0)
        self.start_factor = start_factor
        self.warmup_factor = warmup_factor
        self.end_factor = end_factor

        for group in optimizer.param_groups:
            if 'initial_lr' not in group:
                group.setdefault('initial_lr', group['lr'])

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        lr_factor = self.get_lr_factor(self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= 0:
            return self.start_factor
        elif epoch <= self.warmup_iters:
            grad = (self.warmup_factor - self.start_factor) / self.warmup_iters
            return epoch * grad + self.start_factor
        elif epoch <= self.total_iters:
            grad = (self.end_factor - self.warmup_factor) / self.total_iters
            return epoch * grad + self.warmup_factor
        return self.end_factor
