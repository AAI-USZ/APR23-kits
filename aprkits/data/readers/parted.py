import re
from collections import defaultdict
from typing import Tuple, Dict

from aprkits.data.readers.base import Reader


class PartedReader(Reader):
    def read(
            self
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
        # handle multi-part input and target files
        input_pattern = re.compile(r'(?<=^\.input-)\S+')
        target_pattern = re.compile(r'(?<=^\.target-)\S+')
        input_file_pattern = re.compile(r'\.input$|\.input-\S+')
        target_file_pattern = re.compile(r'\.target$|\.target-\S+')

        train_input_paths = self._get_dir_content((input_file_pattern, '.train'))
        train_target_paths = self._get_dir_content((target_file_pattern, '.train'))
        valid_input_paths = self._get_dir_content((input_file_pattern, '.valid'))
        valid_target_paths = self._get_dir_content((target_file_pattern, '.valid'))
        test_input_paths = self._get_dir_content((input_file_pattern, '.test'))
        test_target_paths = self._get_dir_content((target_file_pattern, '.test'))
        train_inputs = [
            (
                ([
                     input_pattern.search(sfx).group()
                     for sfx in path.suffixes
                     if input_pattern.search(sfx) is not None
                 ] or ['default'])[0],
                path.read_text(encoding='utf-8')
            )
            for path in train_input_paths
        ]
        train_targets = [
            (
                ([
                     target_pattern.search(sfx).group()
                     for sfx in path.suffixes
                     if target_pattern.search(sfx) is not None
                 ] or ['default'])[0],
                path.read_text(encoding='utf-8')
            )
            for path in train_target_paths
        ]
        valid_inputs = [
            (
                ([
                     input_pattern.search(sfx).group()
                     for sfx in path.suffixes
                     if input_pattern.search(sfx) is not None
                 ] or ['default'])[0],
                path.read_text(encoding='utf-8')
            )
            for path in valid_input_paths
        ]
        valid_targets = [
            (
                ([
                     target_pattern.search(sfx).group()
                     for sfx in path.suffixes
                     if target_pattern.search(sfx) is not None
                 ] or ['default'])[0],
                path.read_text(encoding='utf-8')
            )
            for path in valid_target_paths
        ]
        test_inputs = [
            (
                ([
                     input_pattern.search(sfx).group()
                     for sfx in path.suffixes
                     if input_pattern.search(sfx) is not None
                 ] or ['default'])[0],
                path.read_text(encoding='utf-8')
            )
            for path in test_input_paths
        ]
        test_targets = [
            (
                ([
                     target_pattern.search(sfx).group()
                     for sfx in path.suffixes
                     if target_pattern.search(sfx) is not None
                 ] or ['default'])[0],
                path.read_text(encoding='utf-8')
            )
            for path in test_target_paths
        ]

        input_train_items = defaultdict(str)
        target_train_items = defaultdict(str)
        input_valid_items = defaultdict(str)
        target_valid_items = defaultdict(str)
        input_test_items = defaultdict(str)
        target_test_items = defaultdict(str)

        for k_src, src in train_inputs:
            input_train_items[k_src] = '\n'.join((input_train_items[k_src], src)).strip()
        for k_tgt, tgt in train_targets:
            target_train_items[k_tgt] = '\n'.join((target_train_items[k_tgt], tgt)).strip()
        for k_src, src in valid_inputs:
            input_valid_items[k_src] = '\n'.join((input_valid_items[k_src], src)).strip()
        for k_tgt, tgt in valid_targets:
            target_valid_items[k_tgt] = '\n'.join((target_valid_items[k_tgt], tgt)).strip()
        for k_src, src in test_inputs:
            input_test_items[k_src] = '\n'.join((input_test_items[k_src], src)).strip()
        for k_tgt, tgt in test_targets:
            target_test_items[k_tgt] = '\n'.join((target_test_items[k_tgt], tgt)).strip()

        return (
            dict(input_train_items), dict(target_train_items),
            dict(input_valid_items), dict(target_valid_items),
            dict(input_test_items), dict(target_test_items)
        )
