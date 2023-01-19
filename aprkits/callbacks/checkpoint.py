import json
import operator
from datetime import timedelta
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Callable, Any

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


class StatefulModelCheckpoint(ModelCheckpoint):
    def __init__(
            self,
            dirpath: Optional[Union[str, PathLike]] = None,
            filename: Optional[str] = None,
            monitor: Optional[str] = None,
            verbose: bool = False,
            save_last: Optional[bool] = None,
            save_top_k: int = 1,
            save_weights_only: bool = False,
            mode: str = 'min',
            auto_insert_metric_name: bool = True,
            every_n_train_steps: Optional[int] = None,
            train_time_interval: Optional[timedelta] = None,
            every_n_epochs: Optional[int] = None,
            save_on_train_epoch_end: Optional[bool] = None,
            score_fn: Optional[Callable[[Any, Any], bool]] = operator.le
    ):
        super().__init__(
            dirpath, filename, monitor, verbose, save_last, save_top_k, save_weights_only, mode,
            auto_insert_metric_name, every_n_train_steps, train_time_interval, every_n_epochs,
            save_on_train_epoch_end)
        self.best_epoch: Optional[int] = None
        self.current_epoch: Optional[int] = None
        self._state_binary_path = Path(self.dirpath, 'ckpt-state.bin')
        self._score_fn = score_fn if score_fn is not None else operator.le

    def state_dict(self):
        return {
            **super().state_dict(),
            'best_epoch': self.best_epoch,
            'current_epoch': self.current_epoch
        }

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.current_epoch = trainer.current_epoch
        if self._score_fn(self.current_score, self.best_model_score):
            self.best_epoch = trainer.current_epoch
        if not Path(self.dirpath).exists():
            Path(self.dirpath).mkdir(parents=True)
        with open(self._state_binary_path, 'wb') as fp:
            torch.save(self.state_dict(), fp)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.best_epoch = state_dict['best_epoch']
        self.current_epoch = state_dict['current_epoch']

    def restore(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if self._state_binary_path.exists():
            print('Previous checkpoint found. Loading state . . .')
            with open(self._state_binary_path, 'rb') as fp:
                state = torch.load(fp)
                self.load_state_dict(state)
        elif verbose:
            print('No checkpoint state found. Nothing to restore . . .')


class CheckpointState(Callback):
    def __init__(self, checkpoint: ModelCheckpoint):
        self.ckpt = checkpoint
        self.state_path = Path(self.ckpt.dirpath, 'state')
        self._ckpt_name = 'ckpt-state.pk'
        self._info_name = 'info.json'

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            if not self.state_path.exists():
                self.state_path.mkdir(parents=True)
            with open(self.state_path.joinpath(self._ckpt_name), 'wb') as f:
                torch.save(self.ckpt.state_dict(), f)
            with open(self.state_path.joinpath(self._info_name), 'w') as f:
                json.dump({'epoch': trainer.current_epoch}, f)

    def restore_checkpoint(self):
        ckpt_state_path = self.state_path.joinpath(self._ckpt_name)
        info_path = self.state_path.joinpath(self._info_name)

        if ckpt_state_path.exists() and info_path.exists():
            with open(ckpt_state_path, 'rb') as fpa, open(info_path, 'r') as fpb:
                self.ckpt.load_state_dict(torch.load(fpa))
                return self.ckpt, json.load(fpb)
        elif ckpt_state_path.exists() and not info_path.exists():
            with open(ckpt_state_path, 'rb') as fpa:
                self.ckpt.load_state_dict(torch.load(fpa))
                return self.ckpt, None
        return self.ckpt, None
