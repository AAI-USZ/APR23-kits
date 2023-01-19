import itertools
from typing import Tuple, List, Dict

import numpy as np

from aprkits.data.providers import DataProvider


def _split_dict_elements_to_lines(dct):
    return {
        k: v.splitlines()
        for k, v in dct.items()
    }


def _convert_dict_elements_to_numpy(dct, dtype=object):
    return {
        k: np.array(v, dtype=dtype)
        for k, v in dct.items()
    }


def _shuffle_dict_elements_by_indices(dct, indices):
    return {
        k: v[indices]
        for k, v in dct.items()
    }


def _convert_dict_elements_to_list(dct):
    return {
        k: v.tolist()
        for k, v in dct.items()
    }


def _slice_dict_elements(dct, slc):
    return {
        k: v[slc]
        for k, v in dct.items()
    }


class TrainValidTestDataProvider(DataProvider):
    def load_data(
            self
    ) -> Tuple[
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]]
    ]:
        data = self.reader.read()

        if self._is_split:
            inp_train, tar_train, inp_valid, tar_valid, inp_test, tar_test = data

            inp_train = _split_dict_elements_to_lines(inp_train)
            tar_train = _split_dict_elements_to_lines(tar_train)
            inp_valid = _split_dict_elements_to_lines(inp_valid)
            tar_valid = _split_dict_elements_to_lines(tar_valid)
            inp_test = _split_dict_elements_to_lines(inp_test)
            tar_test = _split_dict_elements_to_lines(tar_test)

            assert inp_train.keys() == inp_valid.keys() == inp_test.keys(), \
                'Input train-, validation- and test groups should contain same keys.'
            assert tar_train.keys() == tar_valid.keys() == tar_test.keys(), \
                'Target train-, validation- and test groups should contain same keys.'

            for k1, k2 in itertools.combinations(inp_train.keys(), r=2):
                assert len(inp_train[k1]) == len(inp_train[k2]), \
                    'Different train groups should contain the same amount of data.'
                assert len(inp_valid[k1]) == len(inp_valid[k2]), \
                    'Different validation groups should contain the same amount of data.'
                assert len(inp_test[k1]) == len(inp_test[k2]), \
                    'Different test groups should contain the same amount of data.'

            for k1, k2 in itertools.combinations(tar_train.keys(), r=2):
                assert len(tar_train[k1]) == len(tar_train[k2]), \
                    'Different train groups should contain the same amount of data.'
                assert len(tar_valid[k1]) == len(tar_valid[k2]), \
                    'Different validation groups should contain the same amount of data.'
                assert len(tar_test[k1]) == len(tar_test[k2]), \
                    'Different test groups should contain the same amount of data.'

            # based on previous asserts this check is enough for all data dictionaries
            assert len(inp_train[list(inp_train.keys())[0]]) == len(tar_train[list(tar_train.keys())[0]]), \
                'Input and target should be of the same size.'
            assert len(inp_valid[list(inp_valid.keys())[0]]) == len(tar_valid[list(tar_valid.keys())[0]]), \
                'Input and target should be of the same size.'
            assert len(inp_test[list(inp_test.keys())[0]]) == len(tar_test[list(tar_test.keys())[0]]), \
                'Input and target should be of the same size.'

            if self._shuffle:
                if self._shuffle_rand_seed is not None:
                    np.random.seed(self._shuffle_rand_seed)

                inp_train = _convert_dict_elements_to_numpy(inp_train)
                tar_train = _convert_dict_elements_to_numpy(tar_train)
                inp_valid = _convert_dict_elements_to_numpy(inp_valid)
                tar_valid = _convert_dict_elements_to_numpy(tar_valid)
                inp_test = _convert_dict_elements_to_numpy(inp_test)
                tar_test = _convert_dict_elements_to_numpy(tar_test)

                _train_sample = (
                    inp_train['default'] if 'default' in inp_train else inp_train[list(inp_train.keys())[0]])
                train_indices = np.random.choice(
                    np.shape(_train_sample)[0],
                    size=np.shape(_train_sample)[0],
                    replace=False)
                _valid_sample = (
                    inp_valid['default'] if 'default' in inp_valid else inp_valid[list(inp_valid.keys())[0]])
                valid_indices = np.random.choice(
                    np.shape(_valid_sample)[0],
                    size=np.shape(_valid_sample)[0],
                    replace=False)
                _test_sample = (
                    inp_test['default'] if 'default' in inp_test else inp_test[list(inp_test.keys())[0]])
                test_indices = np.random.choice(
                    np.shape(_test_sample)[0],
                    size=np.shape(_test_sample)[0],
                    replace=False)

                inp_train = _shuffle_dict_elements_by_indices(inp_train, train_indices)
                tar_train = _shuffle_dict_elements_by_indices(tar_train, train_indices)
                inp_valid = _shuffle_dict_elements_by_indices(inp_valid, valid_indices)
                tar_valid = _shuffle_dict_elements_by_indices(tar_valid, valid_indices)
                inp_test = _shuffle_dict_elements_by_indices(inp_test, test_indices)
                tar_test = _shuffle_dict_elements_by_indices(tar_test, test_indices)

                inp_train = _convert_dict_elements_to_list(inp_train)
                tar_train = _convert_dict_elements_to_list(tar_train)
                inp_valid = _convert_dict_elements_to_list(inp_valid)
                tar_valid = _convert_dict_elements_to_list(tar_valid)
                inp_test = _convert_dict_elements_to_list(inp_test)
                tar_test = _convert_dict_elements_to_list(tar_test)
        else:
            inp, tar = data
            inp, tar = _split_dict_elements_to_lines(inp), _split_dict_elements_to_lines(tar)

            for k1, k2 in itertools.combinations(inp.keys(), r=2):
                assert len(inp[k1]) == len(inp[k2]), 'Different input groups should contain the same amount of data.'

            for k1, k2 in itertools.combinations(tar.keys(), r=2):
                assert len(tar[k1]) == len(tar[k2]), 'Different target groups should contain the same amount of data.'

            # based on previous asserts this check is enough for data dictionaries
            assert len(inp[list(inp.keys())[0]]) == len(tar[list(tar.keys())[0]]), \
                'Input and target should be of the same size.'

            if self._shuffle:
                if self._shuffle_rand_seed is not None:
                    np.random.seed(self._shuffle_rand_seed)

                inp = _convert_dict_elements_to_numpy(inp)
                tar = _convert_dict_elements_to_numpy(tar)
                _sample = inp['default'] if 'default' in inp else inp[list(inp.keys())[0]]
                indices = np.random.choice(
                    np.shape(_sample)[0], size=np.shape(_sample)[0], replace=False)
                inp = _shuffle_dict_elements_by_indices(inp, indices)
                tar = _shuffle_dict_elements_by_indices(tar, indices)
                inp = _convert_dict_elements_to_list(inp)
                tar = _convert_dict_elements_to_list(tar)

            train_perc = self._train_valid_test_ratio[0]

            try:
                valid_perc = self._train_valid_test_ratio[1]
            except IndexError:
                valid_perc = 1.0 - train_perc

            i1 = int(len(inp) * train_perc)
            i2 = int(len(inp) * valid_perc + i1)

            inp_train = _slice_dict_elements(inp, slice(i1))
            tar_train = _slice_dict_elements(tar, slice(i1))
            inp_valid = _slice_dict_elements(inp, slice(i1, i2))
            tar_valid = _slice_dict_elements(tar, slice(i1, i2))
            inp_test = _slice_dict_elements(inp, slice(i2, None))
            tar_test = _slice_dict_elements(tar, slice(i2, None))

        return inp_train, tar_train, inp_valid, tar_valid, inp_test, tar_test
