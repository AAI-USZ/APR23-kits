from typing import Iterable


def get_default(val, default_if_none):
    return val if val is not None else default_if_none


def safe_del(dictionary: dict, key: str):
    if key in dictionary:
        del dictionary[key]


def safe_del_keys_from_all(*dictionaries: dict, keys: Iterable[str]):
    for key in keys:
        for dct in dictionaries:
            safe_del(dct, key)
