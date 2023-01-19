import json
import sys
from pathlib import Path

import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.tensorboard import SummaryWriter

from aprkits.utils import json_dict_to_md


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, torch.dtype):
            return str(o)
        return super().default(o)


class ModelParamsState(Callback):
    def __init__(self, summary_writer: SummaryWriter, params: dict):
        self.params = params
        self.summarizer = summary_writer

    def on_sanity_check_end(self, trainer, pl_module):
        if not Path(self.summarizer.log_dir, '.modelsummarycheck').exists():
            summary = ModelSummary(pl_module)
            json_dict = {
                'total_parameters': summary.total_parameters,
                'trainable_parameters': summary.trainable_parameters,
                'model_size': summary.model_size,
                'params': self.params
            }
            json_str = json.dumps(json_dict, indent=2, cls=CustomEncoder)
            md = json_dict_to_md(json_dict)

            interrupted = False
            try:
                self.summarizer.add_text('model summary/json', json_str)
            except KeyboardInterrupt:
                sys.stdout.write('Detected KeyboardInterrupt, attempting graceful shutdown...')
                interrupted = True
                self.summarizer.add_text('model summary/json', json_str)

            try:
                self.summarizer.add_text('model summary/md', md)
            except KeyboardInterrupt:
                sys.stdout.write('Detected KeyboardInterrupt, attempting graceful shutdown...')
                interrupted = True
                self.summarizer.add_text('model summary/md', md)

            try:
                with open(Path(self.summarizer.log_dir, '.modelsummarycheck'), 'w'):
                    pass
            except KeyboardInterrupt:
                sys.stdout.write('Detected KeyboardInterrupt, attempting graceful shutdown...')
                interrupted = True
                with open(Path(self.summarizer.log_dir, '.modelsummarycheck'), 'w'):
                    pass

            if interrupted:
                exit(-1)
