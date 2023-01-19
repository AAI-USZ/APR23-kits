from pathlib import Path
from typing import List, TypedDict, Optional

from typing_extensions import NotRequired

K_IS_SPLIT = 'is_split'
K_SHUFFLE = 'shuffle'
K_PATH = 'path'
K_MAX_LENGTH = 'max_length'
K_MAX_LENGTH_PAIR = 'max_length_pair'
K_TOKENIZER = 'tokenizer'
K_TOKENIZER_PAIR = 'tokenizer_pair'
K_SHUFFLE_RAND_SEED = 'shuffle_rand_seed'
K_TRAIN_VALID_TEST_RATION = 'train_valid_test_ratio'
K_TENSOR_TYPE = 'tensor_type'


DataConfig = TypedDict('DataConfig', {
    'is_split': bool,
    'shuffle': bool,
    'path': Path,
    'max_length': NotRequired[Optional[int]],
    'max_length_pair': NotRequired[Optional[int]],
    'tokenizer': NotRequired[Optional[str]],
    'tokenizer_pair': NotRequired[Optional[str]],
    'shuffle_rand_seed': NotRequired[Optional[int]],
    'train_valid_test_ratio': NotRequired[Optional[List[float]]],
    'tensor_type': NotRequired[Optional[str]],
    'src_line_processor': NotRequired[dict],
    'tgt_line_processor': NotRequired[dict],
})
