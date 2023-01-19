import pickle
import sys
from os import PathLike
from pathlib import Path
from typing import Union, Dict, Tuple, Literal, overload

import numpy as np
from transformers import PreTrainedTokenizer, BatchEncoding

from .providers import TrainValidTestDataProvider
from .utils import get_longer_seq_mask, mask_data


def _select_value_from_dict_by_key_if_present(dct, key):
    return (
        dct[key] if key in dct
        else dct['default'] if 'default' in dct
        else dct[list(dct.keys())[0]])


def _tokenize_dict_elements_with(dct, tokenizer):
    return {
        k: _select_value_from_dict_by_key_if_present(tokenizer, k)(v)
        for k, v in dct.items()
    }


def _get_longer_seq_mask_for_dict_batch_encodings(dct, max_length):
    return {
        k: get_longer_seq_mask(
            v.data['input_ids'], _select_value_from_dict_by_key_if_present(max_length, k)
        )
        for k, v in dct.items()
    }


def _pad_batch_encodings_with_tokenizer(dct, tokenizer, max_length, return_tensors):
    return {
        k: _select_value_from_dict_by_key_if_present(tokenizer, k).pad(
            v,
            padding='max_length',
            max_length=_select_value_from_dict_by_key_if_present(max_length, k),
            return_tensors=return_tensors
        )
        for k, v in dct.items()
    }


def _id_suffix_or_default(suffixes):
    for sx in suffixes:
        if sx.startswith('.&'):
            return sx.lstrip('.&')
    return 'default'


def _is_file_name_ok(path):
    return path.name.split('.')[0] == 'batch_encodings'


def _contains_all(path, suffixes):
    return all(sx in path.suffixes for sx in suffixes)


def _try_load_batch_encoding_part_or_default(
        base_path: Union[str, PathLike],
        mode: Literal['train', 'valid', 'test'],
        src_type: Literal['input', 'target'],
        part: str
):
    paths = [
        path
        for path in Path(base_path).iterdir()
        if _is_file_name_ok(path) and _contains_all(path, [f'.{mode}', f'.{src_type}', f'.&{part}', '.pk'])
    ]
    if len(paths) <= 0:
        paths = [
            path
            for path in Path(base_path).iterdir()
            if _is_file_name_ok(path) and _contains_all(path, [f'.{mode}', f'.{src_type}', '.pk'])
        ]
    path = paths[0]
    return load_batch_encoding_part(path)


def _load_batch_encoding_x_parts(
        base_path: Union[str, PathLike],
        mode: Literal['train', 'valid', 'test'],
        src_type: Literal['input', 'target']
) -> Dict[str, BatchEncoding]:
    paths = [
        (_id_suffix_or_default(path.suffixes), path)
        for path in Path(base_path).iterdir()
        if _is_file_name_ok(path) and _contains_all(path, [f'.{mode}', f'.{src_type}', '.pk'])
    ]
    return {
        new_key: load_batch_encoding_part(path)
        for new_key, path in paths
    }


def _save_batch_encoding_x_parts(
        data: Dict[str, BatchEncoding],
        base_path: Union[str, PathLike],
        mode: Literal['train', 'valid', 'test'],
        src_type: Literal['input', 'target']
):
    for key in data:
        save_batch_encoding_part(
            data[key],
            Path(base_path).joinpath(
                f'batch_encodings.{mode}.{src_type}{(".&" + key) if key != "default" else ""}.pk'
            )
        )


def get_train_valid_test_data(
        *,
        data_provider: TrainValidTestDataProvider,
        tokenizer: Union[PreTrainedTokenizer, Dict[str, PreTrainedTokenizer]],
        tokenizer_pair: Union[PreTrainedTokenizer, Dict[str, PreTrainedTokenizer]] = None,
        max_length: Union[int, Dict[str, int]] = None,
        max_length_pair: Union[int, Dict[str, int]] = None,
        return_tensors: str = None
) -> Tuple[
    Dict[str, BatchEncoding],
    Dict[str, BatchEncoding],
    Dict[str, BatchEncoding],
    Dict[str, BatchEncoding],
    Dict[str, BatchEncoding],
    Dict[str, BatchEncoding]
]:
    if not isinstance(tokenizer, dict):
        tokenizer = {'default': tokenizer}
    if tokenizer_pair is not None and not isinstance(tokenizer_pair, dict):
        tokenizer_pair = {'default': tokenizer_pair}

    if max_length is not None and not isinstance(max_length, dict):
        max_length = {'default': max_length}
    if max_length_pair is not None and not isinstance(max_length_pair, dict):
        max_length_pair = {'default': max_length_pair}

    if tokenizer_pair is None:
        tokenizer_pair = tokenizer
    if max_length is None:
        max_length = {k: tzer.model_max_length for k, tzer in tokenizer}
    if max_length_pair is None:
        max_length_pair = {k: tzer.model_max_length for k, tzer in tokenizer_pair}

    src_train, tgt_train, src_valid, tgt_valid, src_test, tgt_test = data_provider.load()

    # tokenize every data part with its own tokenizer
    src_train_be = _tokenize_dict_elements_with(src_train, tokenizer)
    tgt_train_be = _tokenize_dict_elements_with(tgt_train, tokenizer_pair)
    src_valid_be = _tokenize_dict_elements_with(src_valid, tokenizer)
    tgt_valid_be = _tokenize_dict_elements_with(tgt_valid, tokenizer_pair)
    src_test_be = _tokenize_dict_elements_with(src_test, tokenizer)
    tgt_test_be = _tokenize_dict_elements_with(tgt_test, tokenizer_pair)

    src_train_mask = _get_longer_seq_mask_for_dict_batch_encodings(src_train_be, max_length)
    tgt_train_mask = _get_longer_seq_mask_for_dict_batch_encodings(tgt_train_be, max_length_pair)
    src_valid_mask = _get_longer_seq_mask_for_dict_batch_encodings(src_valid_be, max_length)
    tgt_valid_mask = _get_longer_seq_mask_for_dict_batch_encodings(tgt_valid_be, max_length_pair)
    src_test_mask = _get_longer_seq_mask_for_dict_batch_encodings(src_test_be, max_length)
    tgt_test_mask = _get_longer_seq_mask_for_dict_batch_encodings(tgt_test_be, max_length_pair)

    train_mask = np.logical_and.reduce((*src_train_mask.values(), *tgt_train_mask.values()))
    valid_mask = np.logical_and.reduce((*src_valid_mask.values(), *tgt_valid_mask.values()))
    test_mask = np.logical_and.reduce((*src_test_mask.values(), *tgt_test_mask.values()))

    for be in src_train_be.values():
        be.data = mask_data(be.data, train_mask)
    for be in tgt_train_be.values():
        be.data = mask_data(be.data, train_mask)
    for be in src_valid_be.values():
        be.data = mask_data(be.data, valid_mask)
    for be in tgt_valid_be.values():
        be.data = mask_data(be.data, valid_mask)
    for be in src_test_be.values():
        be.data = mask_data(be.data, test_mask)
    for be in tgt_test_be.values():
        be.data = mask_data(be.data, test_mask)

    src_train_be = _pad_batch_encodings_with_tokenizer(src_train_be, tokenizer, max_length, return_tensors)
    tgt_train_be = _pad_batch_encodings_with_tokenizer(tgt_train_be, tokenizer_pair, max_length_pair, return_tensors)
    src_valid_be = _pad_batch_encodings_with_tokenizer(src_valid_be, tokenizer, max_length, return_tensors)
    tgt_valid_be = _pad_batch_encodings_with_tokenizer(tgt_valid_be, tokenizer_pair, max_length_pair, return_tensors)
    src_test_be = _pad_batch_encodings_with_tokenizer(src_test_be, tokenizer, max_length, return_tensors)
    tgt_test_be = _pad_batch_encodings_with_tokenizer(tgt_test_be, tokenizer_pair, max_length_pair, return_tensors)

    return (
        src_train_be, tgt_train_be,
        src_valid_be, tgt_valid_be,
        src_test_be, tgt_test_be
    )


def save_batch_encoding_part(data: BatchEncoding, path: Union[str, PathLike]):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)


def save_batch_encoding_train_input_parts(data: Dict[str, BatchEncoding], base_path: Union[str, PathLike]):
    _save_batch_encoding_x_parts(data, base_path, 'train', 'input')


def save_batch_encoding_train_target_parts(data: Dict[str, BatchEncoding], base_path: Union[str, PathLike]):
    _save_batch_encoding_x_parts(data, base_path, 'train', 'target')


def save_batch_encoding_valid_input_parts(data: Dict[str, BatchEncoding], base_path: Union[str, PathLike]):
    _save_batch_encoding_x_parts(data, base_path, 'valid', 'input')


def save_batch_encoding_valid_target_parts(data: Dict[str, BatchEncoding], base_path: Union[str, PathLike]):
    _save_batch_encoding_x_parts(data, base_path, 'valid', 'target')


def save_batch_encoding_test_input_parts(data: Dict[str, BatchEncoding], base_path: Union[str, PathLike]):
    _save_batch_encoding_x_parts(data, base_path, 'test', 'input')


def save_batch_encoding_test_target_parts(data: Dict[str, BatchEncoding], base_path: Union[str, PathLike]):
    _save_batch_encoding_x_parts(data, base_path, 'test', 'target')


def save_batch_encodings(
        data: Tuple[
            Dict[str, BatchEncoding],
            Dict[str, BatchEncoding],
            Dict[str, BatchEncoding],
            Dict[str, BatchEncoding],
            Dict[str, BatchEncoding],
            Dict[str, BatchEncoding]
        ],
        base_path: Union[str, PathLike]
):
    base_path = Path(base_path)
    if not base_path.exists():
        base_path.mkdir(parents=True)

    functors = (
        save_batch_encoding_train_input_parts, save_batch_encoding_train_target_parts,
        save_batch_encoding_valid_input_parts, save_batch_encoding_valid_target_parts,
        save_batch_encoding_test_input_parts, save_batch_encoding_test_target_parts)
    for f, d in zip(functors, data):
        try:
            f(d, base_path)
        except IOError as e:
            sys.stderr.write(str(e))


def load_batch_encoding_part(path: Union[str, PathLike]) -> BatchEncoding:
    with open(path, 'rb') as fp:
        return pickle.load(fp)


@overload
def load_batch_encoding_train_part(
        base_path: Union[str, PathLike], part: Literal['<ALL>'] = None
) -> Tuple[Dict[str, BatchEncoding], Dict[str, BatchEncoding]]: ...


@overload
def load_batch_encoding_train_part(
        base_path: Union[str, PathLike], part: str
) -> Tuple[BatchEncoding, BatchEncoding]: ...


def load_batch_encoding_train_part(base_path, part=None):
    if part is None or part == '<ALL>':
        return load_train_batch_encodings(base_path)
    return (
        _try_load_batch_encoding_part_or_default(base_path, 'train', 'input', part),
        _try_load_batch_encoding_part_or_default(base_path, 'train', 'target', part)
    )


@overload
def load_batch_encoding_valid_part(
        base_path: Union[str, PathLike], part: Literal['<ALL>'] = None
) -> Tuple[Dict[str, BatchEncoding], Dict[str, BatchEncoding]]: ...


@overload
def load_batch_encoding_valid_part(
        base_path: Union[str, PathLike], part: str
) -> Tuple[BatchEncoding, BatchEncoding]: ...


def load_batch_encoding_valid_part(base_path, part=None):
    if part is None or part == '<ALL>':
        return load_valid_batch_encodings(base_path)
    return (
        _try_load_batch_encoding_part_or_default(base_path, 'valid', 'input', part),
        _try_load_batch_encoding_part_or_default(base_path, 'valid', 'target', part)
    )


@overload
def load_batch_encoding_test_part(
        base_path: Union[str, PathLike], part: Literal['<ALL>'] = None
) -> Tuple[Dict[str, BatchEncoding], Dict[str, BatchEncoding]]: ...


@overload
def load_batch_encoding_test_part(
        base_path: Union[str, PathLike], part: str
) -> Tuple[BatchEncoding, BatchEncoding]: ...


def load_batch_encoding_test_part(base_path, part=None):
    if part is None or part == '<ALL>':
        return load_test_batch_encodings(base_path)
    return (
        _try_load_batch_encoding_part_or_default(base_path, 'test', 'input', part),
        _try_load_batch_encoding_part_or_default(base_path, 'test', 'target', part)
    )


def load_batch_encoding_mode_part(
        base_path: Union[str, PathLike], mode: Literal['train', 'valid', 'test'], part: str
) -> Tuple[BatchEncoding, BatchEncoding]:
    return (
        _try_load_batch_encoding_part_or_default(base_path, mode, 'input', part),
        _try_load_batch_encoding_part_or_default(base_path, mode, 'target', part)
    )


def load_batch_encoding_train_input_parts(base_path: Union[str, PathLike]):
    return _load_batch_encoding_x_parts(base_path, 'train', 'input')


def load_batch_encoding_train_target_parts(base_path: Union[str, PathLike]):
    return _load_batch_encoding_x_parts(base_path, 'train', 'target')


def load_batch_encoding_valid_input_parts(base_path: Union[str, PathLike]):
    return _load_batch_encoding_x_parts(base_path, 'valid', 'input')


def load_batch_encoding_valid_target_parts(base_path: Union[str, PathLike]):
    return _load_batch_encoding_x_parts(base_path, 'valid', 'target')


def load_batch_encoding_test_input_parts(base_path: Union[str, PathLike]):
    return _load_batch_encoding_x_parts(base_path, 'test', 'input')


def load_batch_encoding_test_target_parts(base_path: Union[str, PathLike]):
    return _load_batch_encoding_x_parts(base_path, 'test', 'target')


def load_train_batch_encodings(base_path: Union[str, PathLike]):
    return load_batch_encoding_train_input_parts(base_path), load_batch_encoding_train_target_parts(base_path)


def load_valid_batch_encodings(base_path: Union[str, PathLike]):
    return load_batch_encoding_valid_input_parts(base_path), load_batch_encoding_valid_target_parts(base_path)


def load_test_batch_encodings(base_path: Union[str, PathLike]):
    return load_batch_encoding_test_input_parts(base_path), load_batch_encoding_test_target_parts(base_path)


def load_batch_encodings(base_path: Union[str, PathLike]):
    return (
        *load_train_batch_encodings(base_path),
        *load_valid_batch_encodings(base_path),
        *load_test_batch_encodings(base_path)
    )
