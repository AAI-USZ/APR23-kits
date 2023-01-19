import json
from pathlib import Path

from aprkits.types import Config, DataConfig


_invalid_conf = False

try:
    _cfg_dict = json.load(open(Path('config.json'), 'r'))
    _data_config_id = _cfg_dict['useData'] if 'useData' in _cfg_dict else 'default'
    _train_config_id = _cfg_dict['useTrain'] if 'useTrain' in _cfg_dict else 'default'
    _data_config = _cfg_dict['dataConfigs'][_data_config_id]
except KeyError or IOError:
    _invalid_conf = True


def get_config():
    if _invalid_conf:
        return None

    return Config(
        data=DataConfig(
            is_split=_data_config['isSplit'],
            shuffle=_data_config['shuffle'],
            path=Path(_data_config['path']),
            max_length=_data_config.get('maxLength'),
            max_length_pair=_data_config.get('maxLengthPair'),
            tokenizer=_data_config.get('tokenizer'),
            tokenizer_pair=_data_config.get('tokenizerPair'),
            shuffle_rand_seed=_data_config.get('shuffleRandSeed'),
            train_valid_test_ratio=_data_config.get('trainValidTestRatio')
        )
    )
