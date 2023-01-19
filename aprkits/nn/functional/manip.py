import torch


def lift_predictions(preds, labels, ignore_index: int = 0, lift: float = 10000.0):
    lifter = torch.zeros(preds.size(), dtype=preds.dtype, device=preds.device)
    lifter[:, ignore_index, :][labels == ignore_index] = lift
    return preds + lifter
