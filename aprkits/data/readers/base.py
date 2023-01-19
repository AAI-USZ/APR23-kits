import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Union

from aprkits.types import DataConfig


class Reader(ABC):
    def __init__(self, path: Path):
        self._path = path

    @classmethod
    def from_config(cls, config: DataConfig):
        return cls(path=config['path'])

    def _get_dir_content(self, suffixes: Iterable[Union[str, re.Pattern]]):
        return [
            path
            for path in self._path.iterdir()
            if path.is_file() and all(
                any(suffix.match(s_sfx) is not None for s_sfx in path.suffixes)
                if isinstance(suffix, re.Pattern)
                else suffix in path.suffixes
                for suffix in suffixes
            )
        ]

    @abstractmethod
    def read(self): ...
