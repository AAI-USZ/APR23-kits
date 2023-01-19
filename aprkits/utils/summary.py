import json
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Summarizer:
    def __init__(self, summary_writer: SummaryWriter):
        self.summarizer = summary_writer

    @staticmethod
    def get_summary_writer(path: Path):
        return SummaryWriter(log_dir=str(path))

    def write_summary_of_scalars(self, step_type='_common_', idx=-1, **kwargs):
        for name, value in kwargs.items():
            name = ' '.join(name.split('_'))
            try:
                if hasattr(value, 'item'):
                    value = value.item()
                self.summarizer.add_scalar(f'{step_type}/{name}', value, global_step=idx)
            except IOError as e:
                sys.stderr.write(str(e))

    def write_model_json_summary(self, json_dict: dict):
        try:
            self.summarizer.add_text('model config', json.dumps(json_dict, indent=2))
        except IOError as e:
            sys.stderr.write(str(e))
