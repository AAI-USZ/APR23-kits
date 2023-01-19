from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict

from aprkits.data.processing import StringProcessor
from aprkits.data.readers import BulkReader, PartedReader, Reader
from aprkits.types import DataConfig


def _iter_apply_on_str_impl(o, fn, iterable=None, key=None):
    if isinstance(o, str):
        if iterable is not None and not isinstance(iterable, tuple):
            iterable[key] = fn(o)
    elif isinstance(o, dict):
        for k, v in o.items():
            _iter_apply_on_str_impl(v, fn, o, k)
    elif isinstance(o, (tuple, list)):
        for i, v in enumerate(o):
            _iter_apply_on_str_impl(v, fn, o, i)


def _iter_apply_on_str(o, fn):
    if isinstance(o, str):
        return fn(o)
    _iter_apply_on_str_impl(o, fn)
    return o


class DataProvider(ABC):
    def __init__(
            self,
            is_split: bool,
            shuffle: bool,
            path: Path,
            shuffle_rand_seed: int = None,
            train_valid_test_ratio: List[float] = None,
            src_line_processor: StringProcessor = None,
            tgt_line_processor: StringProcessor = None
    ):
        self._is_split = is_split
        self._shuffle = shuffle
        self._shuffle_rand_seed = shuffle_rand_seed
        self._train_valid_test_ratio = train_valid_test_ratio
        self._src_line_processor = src_line_processor
        self._tgt_line_processor = tgt_line_processor
        self._reader = (PartedReader if is_split else BulkReader)(path)

    @classmethod
    def from_config(cls, config: DataConfig):
        return cls(
            is_split=config['is_split'],
            shuffle=config['shuffle'],
            path=config['path'],
            shuffle_rand_seed=config['shuffle'],
            train_valid_test_ratio=config['train_valid_test_ratio'],
            src_line_processor=StringProcessor.from_config(config.get('src_line_processor')),
            tgt_line_processor=StringProcessor.from_config(config.get('tgt_line_processor'))
        )

    def load(self):
        data = self.load_data()
        if self._src_line_processor is not None:
            for o in data[::2]:
                _iter_apply_on_str(o, self._src_line_processor)
        if self._tgt_line_processor is not None:
            for o in data[1::2]:
                _iter_apply_on_str(o, self._tgt_line_processor)
        return data

    @abstractmethod
    def load_data(self) -> Tuple[
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]]
    ]:
        ...

    @property
    def reader(self) -> Reader:
        return self._reader
