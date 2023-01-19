from abc import ABC, abstractmethod
from typing import List, Optional, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Accuracy, MeanMetric
from transformers import PreTrainedTokenizerBase

import aprkits.nn.functional as f
from aprkits.types import ForwardBatchLMOutput


class AutoPredSaver(ABC, ModelHooks):
    def __init__(
            self,
            save_pred_labels: bool = True,
            save_preds: bool = False
    ):
        super().__init__()
        self._save_pred_labels = save_pred_labels
        self._save_preds = save_preds
        self._pred_labels = torch.tensor([])
        self._preds = torch.tensor([])

    def save_test_preds(self, pred_tensor):
        if self._save_pred_labels:
            labels = pred_tensor.argmax(dim=-1).flatten().to(torch.int32)
            self._pred_labels = torch.concat((self._pred_labels, labels))
        if self._save_preds:
            self._preds = torch.concat((self._preds, pred_tensor.view(-1, pred_tensor.size(-1))))

    def on_test_batch_end(
            self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ):
        self.save_test_preds(outputs['logits'].detach().cpu())


class AutoEpochEndHookForLossAccFullAcc(ABC, ModelHooks):
    tokenizer: PreTrainedTokenizerBase
    current_epoch: int

    train_accuracy: Accuracy
    val_accuracy: Accuracy
    test_accuracy: Accuracy
    train_full_accuracy: Accuracy
    val_full_accuracy: Accuracy
    test_full_accuracy: Accuracy
    train_loss_metric: MeanMetric
    val_loss_metric: MeanMetric
    test_loss_metric: MeanMetric

    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self):
        self.write_summary_of_scalars(
            step_type='train',
            idx=self.current_epoch,
            loss=self.train_loss_metric.compute(),
            accuracy=self.train_accuracy.compute(),
            full_accuracy=self.train_full_accuracy.compute()
        )

    def on_validation_epoch_end(self):
        self.write_summary_of_scalars(
            step_type='validation',
            idx=self.current_epoch,
            loss=self.val_loss_metric.compute(),
            accuracy=self.val_accuracy.compute(),
            full_accuracy=self.val_full_accuracy.compute()
        )

    def on_test_epoch_end(self):
        self.write_summary_of_scalars(
            step_type='test',
            idx=self.current_epoch,
            loss=self.test_loss_metric.compute(),
            accuracy=self.test_accuracy.compute(),
            full_accuracy=self.test_full_accuracy.compute()
        )

    @abstractmethod
    def write_summary_of_scalars(self, *args, **kwargs) -> None: ...


class AutoStepLightningModuleForLM(ABC, pl.LightningModule):
    tokenizer: PreTrainedTokenizerBase
    current_epoch: int

    train_accuracy: Accuracy
    val_accuracy: Accuracy
    test_accuracy: Accuracy
    train_full_accuracy: Accuracy
    val_full_accuracy: Accuracy
    test_full_accuracy: Accuracy
    train_loss_metric: MeanMetric
    val_loss_metric: MeanMetric
    test_loss_metric: MeanMetric

    optimizer_param_groups: List[dict]

    @abstractmethod
    def forward_batch(self, batch, batch_idx, *args, **kwargs) -> ForwardBatchLMOutput: ...

    def training_step(self, train_batch, batch_idx, *args, **kwargs):
        fblm = self.forward_batch(train_batch, batch_idx, *args, **kwargs)
        loss, logits, labels = fblm.loss, fblm.logits, fblm.labels
        if not fblm.class_first:
            logits = logits.transpose(-1, -2)
        loss_ = self.train_loss_metric(loss)
        acc = self.train_accuracy(logits, labels)
        logits = f.lift_predictions(logits, labels, ignore_index=self.tokenizer.pad_token_id)
        f_acc = self.train_full_accuracy(logits, labels)
        self.log('lr', self.optimizer_param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('t.loss', loss_, on_step=True, on_epoch=True, prog_bar=True)
        self.log('t.acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('t.full', f_acc, on_step=True, on_epoch=True, prog_bar=True)
        return fblm.__dict__

    def validation_step(self, val_batch, batch_idx, *args, **kwargs):
        fblm = self.forward_batch(val_batch, batch_idx, *args, **kwargs)
        loss, logits, labels = fblm.loss, fblm.logits, fblm.labels
        if not fblm.class_first:
            logits = logits.transpose(-1, -2)
        loss_ = self.val_loss_metric(loss)
        acc = self.val_accuracy(logits, labels)
        logits = f.lift_predictions(logits, labels, ignore_index=self.tokenizer.pad_token_id)
        f_acc = self.val_full_accuracy(logits, labels)
        self.log('v.loss', loss_, on_step=False, on_epoch=True, prog_bar=True)
        self.log('v.acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('v.full', f_acc, on_step=False, on_epoch=True, prog_bar=True)
        return fblm.__dict__

    def test_step(self, test_batch, batch_idx, *args, **kwargs):
        fblm = self.forward_batch(test_batch, batch_idx, *args, **kwargs)
        loss, logits, labels = fblm.loss, fblm.logits, fblm.labels
        if not fblm.class_first:
            logits = logits.transpose(-1, -2)
        loss_ = self.test_loss_metric(loss)
        acc = self.test_accuracy(logits, labels)
        logits = f.lift_predictions(logits, labels, ignore_index=self.tokenizer.pad_token_id)
        f_acc = self.test_full_accuracy(logits, labels)
        self.log('T.loss', loss_, on_step=False, on_epoch=True, prog_bar=True)
        self.log('T.acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('T.full', f_acc, on_step=False, on_epoch=True, prog_bar=True)
        return fblm.__dict__
