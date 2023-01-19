import inspect
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric

import aprkits.nn.functional as f
from aprkits.utils import Summarizer, rgetattr, set_trainer_epoch


def _metric_filter(module):
    return isinstance(module, Metric)


class AutoEpochEndCallbackForLossAccFullAcc(Callback):
    def __init__(self, summary_writer: SummaryWriter):
        super().__init__()
        self.summarizer = Summarizer(summary_writer)

    def on_train_epoch_end(self, trainer, pl_module):
        self.summarizer.write_summary_of_scalars(
            step_type='train',
            idx=trainer.current_epoch,
            loss=pl_module.train_loss_metric.compute(),
            accuracy=pl_module.train_accuracy.compute(),
            full_accuracy=pl_module.train_full_accuracy.compute()
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = inspect.getmembers(pl_module, _metric_filter)
        self.summarizer.write_summary_of_scalars(
            step_type='validation',
            idx=trainer.current_epoch,
            loss=pl_module.val_loss_metric.compute(),
            accuracy=pl_module.val_accuracy.compute(),
            full_accuracy=pl_module.val_full_accuracy.compute()
        )

    def on_test_epoch_end(self, trainer, pl_module):
        self.summarizer.write_summary_of_scalars(
            step_type='test',
            idx=trainer.current_epoch,
            loss=pl_module.test_loss_metric.compute(),
            accuracy=pl_module.test_accuracy.compute(),
            full_accuracy=pl_module.test_full_accuracy.compute()
        )


class AutoBatchEndForLM(Callback):
    def __init__(
            self,
            ignore_index_attr: str = 'tokenizer.pad_token_id',
            optimizer_param_groups_attr: str = 'optimizer_param_groups'
    ):
        super().__init__()
        self.ignore_index_attr = ignore_index_attr
        self.optimizer_param_groups_attr = optimizer_param_groups_attr

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int
    ):
        loss = outputs['loss']
        logits = outputs['logits']
        labels = outputs['labels']
        if outputs.get('class_first') is not True:
            logits = logits.transpose(-1, -2)
        loss_ = pl_module.train_loss_metric(loss)
        acc = pl_module.train_accuracy(logits, labels)
        logits = f.lift_predictions(logits, labels, ignore_index=rgetattr(pl_module, self.ignore_index_attr))
        f_acc = pl_module.train_full_accuracy(logits, labels)
        pl_module.log(
            'lr', rgetattr(pl_module, self.optimizer_param_groups_attr)[0]['lr'], on_step=False, on_epoch=True,
            prog_bar=True)
        pl_module.log('t.loss', loss_, on_step=True, on_epoch=True, prog_bar=True)
        pl_module.log('t.acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        pl_module.log('t.full', f_acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ):
        loss = outputs['loss']
        logits = outputs['logits']
        labels = outputs['labels']
        if outputs.get('class_first') is not True:
            logits = logits.transpose(-1, -2)
        loss_ = pl_module.val_loss_metric(loss)
        acc = pl_module.val_accuracy(logits, labels)
        logits = f.lift_predictions(logits, labels, ignore_index=rgetattr(pl_module, self.ignore_index_attr))
        f_acc = pl_module.val_full_accuracy(logits, labels)
        pl_module.log('v.loss', loss_, on_step=False, on_epoch=True, prog_bar=True)
        pl_module.log('v.acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        pl_module.log('v.full', f_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ):
        loss = outputs['loss']
        logits = outputs['logits']
        labels = outputs['labels']
        if outputs.get('class_first') is not True:
            logits = logits.transpose(-1, -2)
        loss_ = pl_module.test_loss_metric(loss)
        acc = pl_module.test_accuracy(logits, labels)
        logits = f.lift_predictions(logits, labels, ignore_index=rgetattr(pl_module, self.ignore_index_attr))
        f_acc = pl_module.test_full_accuracy(logits, labels)
        pl_module.log('T.loss', loss_, on_step=False, on_epoch=True, prog_bar=True)
        pl_module.log('T.acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        pl_module.log('T.full', f_acc, on_step=False, on_epoch=True, prog_bar=True)


class BestModelCheckpoint(Callback):
    def __init__(
            self,
            ckpt: ModelCheckpoint
    ):
        self._ckpt = ckpt
        self._fit_ended = False

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._fit_ended = True

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.ckpt.best_model_path != '':
            if self.ckpt.verbose:
                print('Loading model from best model path, before starting the tests . . .')
            pl_module.load_from_checkpoint(self.ckpt.best_model_path)
        if hasattr(self.ckpt, 'best_epoch'):
            print('Setting epoch for when it was best . . .')
            set_trainer_epoch(trainer, self.ckpt.best_epoch)

    @property
    def fit_ended(self):
        return self._fit_ended

    @property
    def ckpt(self):
        return self._ckpt
