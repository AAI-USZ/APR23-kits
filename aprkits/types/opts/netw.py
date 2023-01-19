import re
from collections import UserDict
from dataclasses import dataclass
from typing import List, Optional, Union

from aprkits.utils import to_camel, nameof


def _getattr(cls):
    def attr(c):
        setattr(c, '__getattr__', lambda self, item: self[to_camel(item)])
        return c

    return attr(cls)


def _todict(cls):
    def attr(c):
        setattr(c, 'todict', lambda self: self.data)
        return c
    return attr(cls)


@dataclass
class TreeSitterPathOptions:
    additional_leaf_nodes: List[str] = None
    regex_signal: re.Pattern = None
    get_child_coeffs: bool = None


@dataclass
class GrapherOptions:
    lang: str = 'c'
    flatten_code: bool = True
    chunk_size: int = 2**13


@_getattr
@_todict
class NumberFormatter(UserDict):
    sep: Optional[str]
    left: Optional[str]
    right: Optional[str]

    def __init__(self, sep: str = None, left: str = None, right: str = None):
        super().__init__({
            to_camel(nameof(sep)): sep,
            to_camel(nameof(left)): left,
            to_camel(nameof(right)): right
        })

    def todict(self): ...


@_getattr
@_todict
class CommandSequencerOpts(UserDict):
    delete_token: Optional[str]
    insert_token: Optional[str]
    position_token: Optional[str]
    number_format: Optional[Union[str, NumberFormatter]]
    return_labels: Optional[bool]
    return_numbers: Optional[bool]
    task_label_in: Optional[str]
    task_label_out: Optional[str]
    remove_task_label_before_processing: Optional[bool] = True
    add_task_label_after_processing: Optional[bool] = True

    def __init__(
            self,
            delete_token: str = None,
            insert_token: str = None,
            position_token: str = None,
            number_format: Union[str, NumberFormatter] = None,
            return_labels: bool = None,
            return_numbers: bool = None,
            task_label_in: str = None,
            task_label_out: str = None,
            remove_task_label_before_processing: bool = True,
            add_task_label_after_processing: bool = True
    ):
        super().__init__({
            to_camel(nameof(delete_token)): delete_token,
            to_camel(nameof(insert_token)): insert_token,
            to_camel(nameof(position_token)): position_token,
            to_camel(nameof(number_format)): number_format,
            to_camel(nameof(return_labels)): return_labels,
            to_camel(nameof(return_numbers)): return_numbers,
            to_camel(nameof(task_label_in)): task_label_in,
            to_camel(nameof(task_label_out)): task_label_out,
            to_camel(nameof(remove_task_label_before_processing)): remove_task_label_before_processing,
            to_camel(nameof(add_task_label_after_processing)): add_task_label_after_processing,
        })

    def todict(self): ...


@_getattr
@_todict
class CommandSequenceOneOutput(UserDict):
    command: str
    labels: Optional[str]
    numbers: Optional[str]

    def __init__(
            self,
            command: str = None,
            labels: Optional[str] = None,
            numbers: Optional[str] = None
    ):
        super().__init__({
            to_camel(nameof(command)): command,
            to_camel(nameof(labels)): labels,
            to_camel(nameof(numbers)): numbers
        })

    def todict(self): ...


@_getattr
@_todict
class CommandSequenceManyOutput(UserDict):
    command: List[str]
    labels: Optional[List[str]]
    numbers: Optional[List[str]]

    def __init__(
            self,
            command: List[str] = None,
            labels: Optional[List[str]] = None,
            numbers: Optional[List[str]] = None
    ):
        super().__init__({
            to_camel(nameof(command)): command,
            to_camel(nameof(labels)): labels,
            to_camel(nameof(numbers)): numbers
        })

    def todict(self): ...


@_getattr
@_todict
class CommandSequenceBatchOutput(UserDict):
    command: List[List[str]]
    labels: Optional[List[List[str]]]
    numbers: Optional[List[List[str]]]

    def __init__(
            self,
            command: List[List[str]] = None,
            labels: Optional[List[List[str]]] = None,
            numbers: Optional[List[List[str]]] = None
    ):
        super().__init__({
            to_camel(nameof(command)): command,
            to_camel(nameof(labels)): labels,
            to_camel(nameof(numbers)): numbers
        })

    def todict(self): ...
