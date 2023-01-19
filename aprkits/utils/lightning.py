from datetime import timedelta
from os import PathLike
from pathlib import Path
from typing import Union, Iterable, Optional, List, Dict

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.plugins import PrecisionPlugin, ClusterEnvironment, CheckpointIO, LayerSync
from pytorch_lightning.profiler import Profiler
from pytorch_lightning.strategies import Strategy


class ExtendedTrainer(pl.Trainer):
    def __init__(
            self,
            lit_model: pl.LightningModule,
            checkpoint_callback: ModelCheckpoint,
            logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
            model_path: Union[str, PathLike] = None,
            enable_checkpointing: bool = True,
            callbacks: Union[List[Callback], Callback, None] = None,
            default_root_dir: Optional[str] = None,
            gradient_clip_val: Union[int, float, None] = None,
            gradient_clip_algorithm: Optional[str] = None,
            process_position: int = 0,
            num_nodes: int = 1,
            num_processes: Optional[int] = None,
            devices: Union[List[int], str, int, None] = None,
            gpus: Union[List[int], str, int, None] = None,
            auto_select_gpus: bool = False,
            tpu_cores: Union[List[int], str, int, None] = None,
            ipus: Optional[int] = None,
            log_gpu_memory: Optional[str] = None,
            progress_bar_refresh_rate: Optional[int] = None,
            enable_progress_bar: bool = True,
            overfit_batches: Union[int, float] = 0.0,
            track_grad_norm: Union[int, float, str] = -1,
            check_val_every_n_epoch: int = 1,
            fast_dev_run: Union[int, bool] = False,
            accumulate_grad_batches: Union[int, Dict[int, int], None] = None,
            max_epochs: Optional[int] = None,
            min_epochs: Optional[int] = None,
            max_steps: int = -1,
            min_steps: Optional[int] = None,
            max_time: Union[str, timedelta, Dict[str, int], None] = None,
            limit_train_batches: Union[int, float, None] = None,
            limit_val_batches: Union[int, float, None] = None,
            limit_test_batches: Union[int, float, None] = None,
            limit_predict_batches: Union[int, float, None] = None,
            val_check_interval: Union[int, float, None] = None,
            flush_logs_every_n_steps: Optional[int] = None,
            log_every_n_steps: int = 50,
            accelerator: Union[str, Accelerator, None] = None,
            strategy: Union[str, Strategy, None] = None,
            sync_batchnorm: bool = False,
            precision: Union[int, str] = 32,
            enable_model_summary: bool = True,
            weights_summary: Optional[str] = 'top',
            weights_save_path: Optional[str] = None,
            num_sanity_val_steps: int = 2,
            resume_from_checkpoint: Union[Path, str, None] = None,
            profiler: Union[Profiler, str, None] = None,
            benchmark: Optional[bool] = None,
            deterministic: Optional[bool] = None,
            reload_dataloaders_every_n_epochs: int = 0,
            auto_lr_find: Union[bool, str] = False,
            replace_sampler_ddp: bool = True,
            detect_anomaly: bool = False,
            auto_scale_batch_size: Union[str, bool] = False,
            prepare_data_per_node: Optional[bool] = None,
            plugins: Union[
                Strategy,
                PrecisionPlugin,
                ClusterEnvironment,
                CheckpointIO,
                LayerSync,
                str,
                List[Union[Strategy, PrecisionPlugin, ClusterEnvironment, CheckpointIO, LayerSync, str]],
                None
            ] = None,
            amp_backend: str = 'native',
            amp_level: Optional[str] = None,
            move_metrics_to_cpu: bool = False,
            multiple_trainloader_mode: str = 'max_size_cycle',
            stochastic_weight_avg: bool = False,
            terminate_on_nan: Optional[bool] = None,
    ):
        self._lit_model = load_model_or_checkpoint(
            lit_model=lit_model, checkpoint=checkpoint_callback, model_path=model_path)

        super().__init__(
            logger=logger,
            enable_checkpointing=enable_checkpointing,
            callbacks=callbacks,
            default_root_dir=default_root_dir,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            process_position=process_position,
            num_nodes=num_nodes,
            num_processes=num_processes,
            devices=devices,
            gpus=gpus,
            auto_select_gpus=auto_select_gpus,
            tpu_cores=tpu_cores,
            ipus=ipus,
            log_gpu_memory=log_gpu_memory,
            progress_bar_refresh_rate=progress_bar_refresh_rate,
            enable_progress_bar=enable_progress_bar,
            overfit_batches=overfit_batches,
            track_grad_norm=track_grad_norm,
            check_val_every_n_epoch=check_val_every_n_epoch,
            fast_dev_run=fast_dev_run,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            val_check_interval=val_check_interval,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            strategy=strategy,
            sync_batchnorm=sync_batchnorm,
            precision=precision,
            enable_model_summary=enable_model_summary,
            weights_summary=weights_summary,
            weights_save_path=weights_save_path,
            num_sanity_val_steps=num_sanity_val_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            profiler=profiler,
            benchmark=benchmark,
            deterministic=deterministic,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            auto_lr_find=auto_lr_find,
            replace_sampler_ddp=replace_sampler_ddp,
            detect_anomaly=detect_anomaly,
            auto_scale_batch_size=auto_scale_batch_size,
            prepare_data_per_node=prepare_data_per_node,
            plugins=plugins,
            amp_backend=amp_backend,
            amp_level=amp_level,
            move_metrics_to_cpu=move_metrics_to_cpu,
            multiple_trainloader_mode=multiple_trainloader_mode,
            stochastic_weight_avg=stochastic_weight_avg,
            terminate_on_nan=terminate_on_nan
        )

        if hasattr(checkpoint_callback, 'best_epoch') and checkpoint_callback.best_epoch is not None:
            set_trainer_epoch(self, checkpoint_callback.best_epoch + 1)

    @property
    def loaded_model(self):
        return self._lit_model


def set_trainer_epoch(trainer, epoch: int):
    trainer.fit_loop.epoch_progress.current.completed = epoch
    trainer.fit_loop.epoch_progress.current.processed = epoch
    trainer.fit_loop.epoch_progress.current.ready = epoch
    trainer.fit_loop.epoch_progress.current.started = epoch
    trainer.fit_loop.epoch_progress.total.completed = epoch
    trainer.fit_loop.epoch_progress.total.processed = epoch
    trainer.fit_loop.epoch_progress.total.ready = epoch
    trainer.fit_loop.epoch_progress.total.started = epoch
    return trainer


def load_model_or_checkpoint(
        lit_model: pl.LightningModule,
        checkpoint: ModelCheckpoint,
        model_path: Union[str, PathLike] = None,
        verbose: bool = None
):
    """
    Helps loading a model from path or from checkpoint.
    If both given, the model will load from the model path.

    :param lit_model: model which should handle the checkpoint loading
    :param checkpoint: checkpoint to load from
    :param model_path: model path to load from
    :param verbose: verbosity of the function
    :return: loaded model with the same architecture as the one passed to the function
    """
    if verbose is None:
        verbose = checkpoint.verbose

    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            checkpoint.best_model_path = str(model_path)

    if checkpoint.best_model_path != '':
        if verbose:
            print(f'Loading from checkpoint: {checkpoint.best_model_path}')
        lit_model = lit_model.load_from_checkpoint(checkpoint.best_model_path)

    return lit_model
