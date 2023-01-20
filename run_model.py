import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, MeanMetric
from transformers import T5TokenizerFast, T5ForConditionalGeneration, RobertaModel, RobertaConfig, GPT2TokenizerFast, \
    RobertaTokenizerFast, GPTNeoForCausalLM

import aprkits.nn.functional as f
from aprkits.callbacks import AutoEpochEndCallbackForLossAccFullAcc, \
    AutoBatchEndForLM, BestModelCheckpoint, StatefulModelCheckpoint
from aprkits.data import BatchEncodingDataset
from aprkits.nn import CatGraphNet
from aprkits.optim import CosineWarmupScheduler
from aprkits.tokenizers import NumberTokenizer
from aprkits.utils import set_trainer_epoch, load_model_or_checkpoint


class Args(Namespace):
    max_epochs: int
    warmup_steps: int
    batch_size: int
    learning_rate: float
    use_lr_scheduler: bool
    optim_eps: float
    dropout_rate: float
    accelerator: str
    devices: List[int]
    torch_num_threads: int
    model_max_length: int

    early_stop_monitor: str = 'v.loss'
    early_stop_min_delta: float = 0.1
    early_stop_mode: str = 'min'
    early_stop_patience: int = 8

    limit_train_batches: Optional[int]
    limit_val_batches: Optional[int]
    limit_test_batches: Optional[int]
    limit_pred_batches: Optional[int]

    data_input_dir: str
    tokenizer_cache_dir: Optional[str]
    model_cache_dir: Optional[str]
    summary_dir: str
    ckpt_dir: str
    model_dir: str
    no_model_save: bool
    save_top_k: int

    experiment: str
    representation: str


TIME_STAMP = str(datetime.now().timestamp())

arg_parser = ArgumentParser(
    'Model Trainer Script',
    description='This script is for running different configurations of model fitting.')

arg_parser.add_argument(
    '-i', '--data_input_dir', type=str, required=True,
    help='Specify the data for training.'
         'Input folder should contain .train, .valid and .test files of corresponding data.'
)
arg_parser.add_argument(
    '-X', '--experiment', type=str, choices=['t5', 'codet5', 'graph'], required=True,
    help='Experiment configuration.'
         'Option t5 defines the experiments on t5-base pretrained model, '
         'while codet5 will use Salesforce/codet5-base. '
         'Graph is based on a composite model described in the paper.')
arg_parser.add_argument(
    '-r', '--representation', type=str, choices=['text', 'cmdseqtoken', 'graphtext'], required=True,
    help='Data representation that will be used during training.')

arg_parser.add_argument(
    '-E', '--max_epochs', default=50, type=int, required=False,
    help='Maximum number of epochs the model can train. '
         'Early stopping might kill training before this. '
         'Affects learning rate, if a learning rate scheduler is used (pass -ls or --use_lr_scheduler).')
arg_parser.add_argument(
    '-w', '--warmup_steps', default=1, type=int, required=False,
    help='Specifies warmup steps measured in epochs, until specified learning rate is reached.')
arg_parser.add_argument(
    '-b', '--batch_size', default=8, type=int, required=False,
    help='Specifies the batch size used during forward pass.')
arg_parser.add_argument(
    '-lr', '--learning_rate', default=5e-5, type=float, required=False,
    help='Specify a learning rate. Is a scheduler is used, then this will be the peak of your lr.')
arg_parser.add_argument(
    '-ls', '--use_lr_scheduler', action='store_true', help='Whether to use (linear) learning rate scheduler or not.')
arg_parser.add_argument(
    '-e', '--optim_eps', default=1e-8, type=float, required=False,
    help='Epsilon passed to the optimizer.')
arg_parser.add_argument(
    '-d', '--dropout_rate', default=0.2, type=float, required=False,
    help='Dropout rate used during training.')
arg_parser.add_argument(
    '-a', '--accelerator', default='gpu', choices=['cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'auto'], required=False,
    help='The type of accelerator your model will be running on.')
arg_parser.add_argument(
    '-D', '--devices', nargs='+', default=[0], type=int, required=False,
    help='Visible devices.')
arg_parser.add_argument(
    '-t', '--torch_num_threads', default=6, type=int, required=False,
    help='Number of threads allowed to be used by torch.')
arg_parser.add_argument(
    '-ml', '--model_max_length', default=512, type=int, required=False,
    help='Maximum sequence length of your model. Tokenization will use this number, too.')

arg_parser.add_argument(
    '-em', '--early_stop_monitor', default='v.loss', type=str, choices=['v.loss', 'v.acc', 'v.full'], required=False,
    help='Metric to monitor during early stopping.')
arg_parser.add_argument(
    '-ed', '--early_stop_min_delta', default=0.05, type=float, required=False,
    help='Minimum change required on watched metric, during patience time.')
arg_parser.add_argument(
    '-eM', '--early_stop_mode', default='min', type=str, choices=['min', 'max'], required=False,
    help='Monitor mode of watched metric.')
arg_parser.add_argument(
    '-ep', '--early_stop_patience', default=8, type=int, required=False,
    help='Patience interval, measured in epochs.')

arg_parser.add_argument(
    '-ltb', '--limit_train_batches', default=None, type=int, required=False,
    help='Set, if you want to limit training batches. Useful for brainstorming.')
arg_parser.add_argument(
    '-lvb', '--limit_val_batches', default=None, type=int, required=False,
    help='Set, if you want to limit validation batches. Useful for brainstorming.')
arg_parser.add_argument(
    '-lTb', '--limit_test_batches', default=None, type=int, required=False,
    help='Set, if you want to limit test batches.')
arg_parser.add_argument(
    '-lpb', '--limit_pred_batches', default=None, type=int, required=False,
    help='Set, if you want to limit prediction batches. Useful when you want to see some quick examples.')

arg_parser.add_argument(
    '-tcD', '--tokenizer_cache_dir', default=None, required=False,
    help='Specifies cache dir that will be used by transformers, when downloading from hub.')
arg_parser.add_argument(
    '-mcD', '--model_cache_dir', default=None, required=False,
    help='Specifies cache dir that will be used by transformers, when downloading from hub.')
arg_parser.add_argument(
    '-sD', '--summary_dir', default=f'data/summary/{TIME_STAMP}', required=False,
    help='Specifies where the summaries should be written.')
arg_parser.add_argument(
    '-pD', '--preds_dir', default=f'data/preds/{TIME_STAMP}', required=False,
    help='Specifies output directory for predictions.')
arg_parser.add_argument(
    '-xD', '--ckpt_dir', default=f'data/checkpoints/{TIME_STAMP}', required=False,
    help='Specifies directory for storing model checkpoints.')
arg_parser.add_argument(
    '-mD', '--model_dir', default=f'data/models/{TIME_STAMP}', required=False,
    help='Specifies directory to store your trained model.')
arg_parser.add_argument(
    '-nS', '--no_model_save', action='store_true',
    help='Pass if you do not want to save the trained model. Checkpoints will still be stored.')

arg_parser.add_argument(
    '-st', '--save_top_k', default=10, type=int, required=False,
    help='Maximum number of checkpoints to be saved. '
         'Advised to be larger than, or equal to patience.')


class LitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.max_epochs = prog_args.max_epochs
        self.warmup = prog_args.warmup_steps
        self.lr = prog_args.learning_rate
        self.optim_eps = prog_args.optim_eps
        self.dropout = prog_args.dropout_rate

        if prog_args.experiment == 't5':
            self.tokenizer = T5TokenizerFast.from_pretrained(
                't5-base', cache_dir=prog_args.tokenizer_cache_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(
                't5-base', cache_dir=Path(prog_args.model_cache_dir).joinpath('t5-base'))

            if prog_args.representation == 'cmdseqtoken':
                self.tokenizer.add_tokens(['</[DEL]/>', '</[INS]/>', '</[LOC]/>'])
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.shared.weight = self.model.shared.weight.requires_grad_(False)
                self.model.shared.weight[-3:, :] = torch.zeros((3, self.model.config.hidden_size))
                self.model.shared.weight = self.model.shared.weight.requires_grad_(True)

        elif prog_args.experiment == 'codet5':
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                'Salesforce/codet5-base', cache_dir=prog_args.tokenizer_cache_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(
                'Salesforce/codet5-base', cache_dir=Path(prog_args.model_cache_dir).joinpath('Salesforce/codet5-base'))

            if prog_args.representation == 'cmdseqtoken':
                self.tokenizer.add_tokens(['</[DEL]/>', '</[INS]/>', '</[LOC]/>'])
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.shared.weight = self.model.shared.weight.requires_grad_(False)
                self.model.shared.weight[-3:, :] = torch.zeros((3, self.model.config.hidden_size))
                self.model.shared.weight = self.model.shared.weight.requires_grad_(True)

        elif prog_args.experiment == 'graph':
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                'EleutherAI/gpt-neo-125M',
                cache_dir=Path(prog_args.tokenizer_cache_dir).joinpath('EleutherAI/gpt-neo-125M'))
            self.tokenizer_ = GPT2TokenizerFast.from_pretrained(
                'microsoft/codebert-base',
                cache_dir=Path(prog_args.tokenizer_cache_dir).joinpath('microsoft/codebert-base'))
            self.graph_tokenizer = RobertaTokenizerFast.from_pretrained(
                'resources/tokenizers/codebert.graph')
            self.type_tokenizer = RobertaTokenizerFast.from_pretrained(
                'resources/tokenizers/codebert.graph.types')
            self.num_tokenizer = NumberTokenizer(base=0)

            self.tokenizer.padding_side = 'right'
            self.tokenizer.pad_token = self.tokenizer.eos_token

            graph_config = RobertaConfig.from_pretrained(
                'roberta-base', cache_dir=Path(prog_args.model_cache_dir).joinpath('roberta-base'))
            graph_config.vocab_size = len(self.graph_tokenizer)
            graph_config.type_vocab_size = len(self.type_tokenizer)
            graph_encoder = RobertaModel(graph_config)
            child_embed = nn.Embedding(100, graph_config.hidden_size)
            token_encoder = RobertaModel.from_pretrained(
                'microsoft/codebert-base',
                cache_dir=Path(prog_args.model_cache_dir).joinpath('microsoft/codebert-base'))
            token_decoder = GPTNeoForCausalLM.from_pretrained(
                'EleutherAI/gpt-neo-125M',
                cache_dir=Path(prog_args.model_cache_dir).joinpath('EleutherAI/gpt-neo-125M'))
            self.model = CatGraphNet(
                graph_encoder=graph_encoder,
                token_encoder=token_encoder,
                child_embed=child_embed,
                token_decoder=token_decoder
            )
        else:
            raise NotImplementedError(f'Sorry, experiments on {prog_args.experiment} is not implemented.')

        self.criterion = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.train_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, mdmc_average='global')
        self.val_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, mdmc_average='global')
        self.test_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, mdmc_average='global')

        self.train_full_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, subset_accuracy=True)
        self.val_full_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, subset_accuracy=True)
        self.test_full_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, subset_accuracy=True)

        self.train_loss_metric = MeanMetric(nan_strategy='ignore')
        self.val_loss_metric = MeanMetric(nan_strategy='ignore')
        self.test_loss_metric = MeanMetric(nan_strategy='ignore')
        self.optimizer_param_groups = [{}]

        self._test_pred_labels = torch.tensor([], dtype=torch.int32, device='cpu')
        self._test_true_labels = torch.tensor([], dtype=torch.int32, device='cpu')

    def _save_test_preds(self, pred_tensor):
        labels = pred_tensor.argmax(dim=-1).flatten().to(torch.int32)
        self._test_pred_labels = torch.concat((self._test_pred_labels, labels))

    def forward(self, *args, **kwargs):
        y = self.model(*args, **kwargs)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=self.optim_eps)

        optim_config = {
            'optimizer': optimizer
        }

        if prog_args.use_lr_scheduler:
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer, warmup_iters=self.warmup, total_iters=self.max_epochs)
            optim_config['lr_scheduler'] = scheduler

        self.optimizer_param_groups = optimizer.param_groups

        return optim_config

    def forward_batch(self, batch, step_type: str = None):
        if prog_args.experiment == 't5' or prog_args.experiment == 'codet5':
            src_data, tgt_data, src_data_mask, _ = batch

            model_output = self(
                input_ids=src_data,
                labels=tgt_data,
                attention_mask=src_data_mask
            )
            logits = model_output.logits
            logits = logits.transpose(-1, -2)
            loss = self.criterion(logits, tgt_data)
            loss_val = self.train_loss_metric(loss)
            acc_val = self.train_accuracy(logits, tgt_data)
            logits = f.lift_predictions(logits, tgt_data, ignore_index=self.tokenizer.pad_token_id)
            full_acc_val = self.train_full_accuracy(logits, tgt_data)

            return {
                'logits': model_output.logits,
                'labels': tgt_data,
                'loss': loss,
                'mean_loss': loss_val,
                'accuracy': acc_val,
                'full_accuracy': full_acc_val,
                'step_type': step_type
            }

        elif prog_args.experiment == 'graph':
            (token_input_ids, token_input_attention_mask, node_input_ids, type_input_ids, count_input_ids,
             node_attention_mask, token_target_ids, _) = batch

            labels = token_target_ids
            model_output = self(
                token_input_ids=token_input_ids,
                node_input_ids=node_input_ids,
                type_input_ids=type_input_ids,
                count_input_ids=count_input_ids,
                token_target_ids=token_target_ids,
                token_input_attention_mask=token_input_attention_mask,
                node_attention_mask=node_attention_mask,
            )
            logits = model_output.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits[:, :shift_labels.shape[-1], :]

            shift_logits_ = shift_logits.transpose(-1, -2)
            loss = self.criterion(shift_logits_, shift_labels)

            return {
                'logits': shift_logits,
                'labels': shift_labels,
                'loss': loss,
                'step_type': step_type
            }
        else:
            raise NotImplementedError()

    def training_step(self, train_batch, batch_idx, *args, **kwargs):
        out = self.forward_batch(train_batch, 'train')
        return out

    def validation_step(self, val_batch, batch_idx, *args, **kwargs):
        out = self.forward_batch(val_batch, 'validation')
        return out

    def test_step(self, test_batch, batch_idx, *args, **kwargs):
        out = self.forward_batch(test_batch, 'test')
        return out

    def _get_loader(self, dt: str):
        if prog_args.representation == 'text':
            prefx = 'text'
        elif prog_args.representation == 'cmdseqtoken':
            prefx = 'cmdseq'
        elif prog_args.representation == 'graphtext':
            prefx = 'graph+sequence'
        else:
            raise NotImplementedError()

        shuffle = False
        if dt == 'train':
            shuffle = True

        if prog_args.experiment == 'graph' or prog_args.representation == 'graphtext':
            with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.input')) as fp:
                x_content = fp.read()
                x_content = x_content.splitlines()
            with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.input-node')) as fp:
                x_node_content = fp.read()
                x_node_content = x_node_content.splitlines()
            with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.input-type')) as fp:
                x_type_content = fp.read()
                x_type_content = x_type_content.splitlines()
            with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.input-count')) as fp:
                x_count_content = fp.read()
                x_count_content = x_count_content.splitlines()
            with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.target')) as fp:
                y_content = fp.read()
                y_content = y_content.splitlines()

            x = self.tokenizer_(
                x_content, padding='max_length', truncation=True,
                max_length=prog_args.model_max_length, return_tensors='pt')
            x_node = self.graph_tokenizer(
                x_node_content, padding='max_length', truncation=True,
                max_length=prog_args.model_max_length, return_tensors='pt')
            x_type = self.type_tokenizer(
                x_type_content, padding='max_length', truncation=True,
                max_length=prog_args.model_max_length, return_tensors='pt')
            x_count = self.num_tokenizer(
                x_count_content, padding='max_length', truncation=True,
                max_length=prog_args.model_max_length, return_tensors='pt')
            y = self.tokenizer(
                y_content, padding='max_length', truncation=True,
                max_length=prog_args.model_max_length, return_tensors='pt')

            loader = DataLoader(
                TensorDataset(
                    x.data['input_ids'], x.data['attention_mask'], x_node.data['input_ids'], x_type.data['input_ids'],
                    x_count.data['input_ids'], x_node.data['attention_mask'],
                    y.data['input_ids'], y.data['attention_mask']),
                batch_size=prog_args.batch_size,
                shuffle=shuffle,
                num_workers=6
            )
        else:
            postfix = ''
            if prog_args.representation == 'cmdseqtoken':
                postfix = '-label'

            with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.input')) as fp:
                x_content = fp.read()
                x_content = x_content.splitlines()
            with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.target{postfix}')) as fp:
                y_content = fp.read()
                y_content = y_content.splitlines()

            x = self.tokenizer(
                x_content, padding='max_length', truncation=True,
                max_length=prog_args.model_max_length, return_tensors='pt')
            y = self.tokenizer(
                y_content, padding='max_length', truncation=True,
                max_length=prog_args.model_max_length, return_tensors='pt')

            loader = DataLoader(
                BatchEncodingDataset(x, y),
                batch_size=prog_args.batch_size,
                shuffle=shuffle,
                num_workers=6
            )

        print(f'{dt.title()} input shape:    ', x['input_ids'].shape)
        print(f'{dt.title()} target shape:   ', y['input_ids'].shape)

        return loader

    def train_dataloader(self):
        return self._get_loader('train')

    def val_dataloader(self):
        return self._get_loader('valid')

    def test_dataloader(self):
        return self._get_loader('test')


def create_trainer():
    torch.set_num_threads(prog_args.torch_num_threads)
    lit_model = LitModule()

    ckpt_path = Path(prog_args.ckpt_dir)
    root_dir = Path('.torch-lightning')
    summary_dir = Path(prog_args.summary_dir)

    with SummaryWriter(log_dir=str(summary_dir)) as summary_writer:
        checkpoint_callback = StatefulModelCheckpoint(
            save_top_k=10,
            monitor=prog_args.early_stop_monitor,
            mode=prog_args.early_stop_mode,
            dirpath=ckpt_path,
            filename='{epoch:04d}-{v.loss:.2f}-{v.acc:.2f}-{v.full:.2f}',
            save_on_train_epoch_end=True,
            verbose=True
        )
        early_stopping_callback = EarlyStopping(
            monitor=prog_args.early_stop_monitor,
            min_delta=prog_args.early_stop_min_delta,
            mode=prog_args.early_stop_mode,
            patience=prog_args.early_stop_patience,
            verbose=True
        )
        auto_epoch_end_callback = AutoEpochEndCallbackForLossAccFullAcc(summary_writer)
        auto_batch_end_callback = AutoBatchEndForLM()
        best_model_checkpoint_callback = BestModelCheckpoint(ckpt=checkpoint_callback)
        checkpoint_callback.restore(verbose=True)

        trainer = Trainer(
            max_epochs=prog_args.max_epochs,
            devices=prog_args.devices,
            limit_train_batches=prog_args.limit_train_batches,
            limit_val_batches=prog_args.limit_val_batches,
            limit_test_batches=prog_args.limit_test_batches,
            limit_predict_batches=prog_args.limit_pred_batches,
            accelerator=prog_args.accelerator,
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                auto_epoch_end_callback,
                auto_batch_end_callback,
                best_model_checkpoint_callback
            ],
            default_root_dir=str(root_dir)
        )

        if hasattr(checkpoint_callback, 'best_epoch') and checkpoint_callback.best_epoch is not None:
            set_trainer_epoch(trainer, checkpoint_callback.best_epoch + 1)

        lit_model = load_model_or_checkpoint(lit_model=lit_model, checkpoint=checkpoint_callback)

        return trainer, lit_model


def run_trainer():
    trainer, lit_model = create_trainer()
    trainer.fit(model=lit_model)
    trainer.test(model=lit_model)


def save_model():
    trainer, lit_model = create_trainer()
    if prog_args.experiment == 'graph':
        if not Path(prog_args.model_dir).exists():
            Path(prog_args.model_dir).mkdir(parents=True)
        torch.save(lit_model.model.state_dict(), Path(prog_args.model_dir).joinpath('graph.state_dict'))
    else:
        lit_model.model.save_pretrained(prog_args.model_dir)


def main():
    run_trainer()
    if not prog_args.no_model_save:
        save_model()
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        prog_args = arg_parser.parse_args(namespace=Args())
        main()
    else:
        arg_parser.print_help()
