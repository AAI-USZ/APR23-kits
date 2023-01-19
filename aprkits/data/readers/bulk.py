import re
from collections import defaultdict
from typing import Dict, Tuple

from aprkits.data.readers.base import Reader


class BulkReader(Reader):
    def read(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        # handle multi-part input and target files
        input_pattern = re.compile(r'(?<=^\.input-)\S+')
        target_pattern = re.compile(r'(?<=^\.target-)\S+')
        input_paths = self._get_dir_content((re.compile(r'\.input$|\.input-\S+'),))
        target_paths = self._get_dir_content((re.compile(r'\.target$|\.target-\S+'),))
        inputs = [
            (
                ([
                     input_pattern.search(sfx).group()
                     for sfx in path.suffixes
                     if input_pattern.search(sfx) is not None
                 ] or ['default'])[0],
                path.read_text(encoding='utf-8')
            )
            for path in input_paths
        ]
        targets = [
            (
                ([
                     target_pattern.search(sfx).group()
                     for sfx in path.suffixes
                     if target_pattern.search(sfx) is not None
                 ] or ['default'])[0],
                path.read_text(encoding='utf-8')
            )
            for path in target_paths
        ]

        input_items = defaultdict(str)
        target_items = defaultdict(str)

        for k_src, src in inputs:
            input_items[k_src] = '\n'.join((input_items[k_src], src)).strip()
        for k_tgt, tgt in targets:
            target_items[k_tgt] = '\n'.join((target_items[k_tgt], tgt)).strip()

        return dict(input_items), dict(target_items)
