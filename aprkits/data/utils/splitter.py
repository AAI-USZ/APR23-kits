from typing import Tuple

import numpy as np


def train_valid_test_split_str(
        data: str,
        data_pair: str = None,
        train_valid_test_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_str: str = '\n',
        shuffle: bool = True,
        seed: int = None
):
    if shuffle and seed is not None:
        np.random.seed(seed)
    parts = data.split(split_str)

    if data_pair is not None:
        parts_pair = data_pair.split(split_str)
        parts = list(zip(parts, parts_pair))

    if shuffle:
        np.random.shuffle(parts)

    i1 = int(len(parts) * train_valid_test_ratios[0])
    i2 = int(len(parts) * train_valid_test_ratios[1]) + i1

    train_set = parts[:i1]
    valid_set = parts[i1:i2]
    test_set = parts[i2:]

    if data_pair is None:
        train_set = split_str.join(train_set)
        valid_set = split_str.join(valid_set)
        test_set = split_str.join(test_set)
        return train_set, valid_set, test_set

    train_set, train_set_pair = zip(*train_set)
    valid_set, valid_set_pair = zip(*valid_set)
    test_set, test_set_pair = zip(*test_set)
    train_set, train_set_pair = split_str.join(train_set), split_str.join(train_set_pair)
    valid_set, valid_set_pair = split_str.join(valid_set), split_str.join(valid_set_pair)
    test_set, test_set_pair = split_str.join(test_set), split_str.join(test_set_pair)
    return train_set, train_set_pair, valid_set, valid_set_pair, test_set, test_set_pair
