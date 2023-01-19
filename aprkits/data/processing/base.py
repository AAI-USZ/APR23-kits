from typing import Literal, List, overload, Optional

from aprkits.data.processing.abc import _StringProcessor


class StringProcessor(_StringProcessor):
    def __init__(
            self,
            mode: Literal['rm', 'add'],
            pos_type: Literal['prefix', 'postfix'],
            label: str,
            label_glue: Optional[str] = ' ',
            clean_spaces: Optional[bool] = True
    ):
        if label_glue is None:
            label_glue = ' '
        if clean_spaces is None:
            clean_spaces = True

        super().__init__(clean_spaces=clean_spaces)
        available_modes = {'rm', 'add'}
        available_pos_types = {'prefix', 'postfix'}
        assert mode in available_modes, f'mode must be one of {available_modes}'
        assert pos_type in available_pos_types, f'pos_type must be one of {available_pos_types}'
        self._mode = mode
        self._pos_type = pos_type
        self._label = label
        self._label_glue = label_glue

    @classmethod
    def from_config(cls, config: Optional[dict]):
        if config is None:
            return None

        return cls(
            mode=config['mode'],
            pos_type=config['pos_type'],
            label=config['label'],
            label_glue=config.get('label_glue'),
            clean_spaces=config.get('clean_spaces')
        )

    @overload
    def process(self, data: str, *args, **kwargs): ...

    @overload
    def process(self, data: List[str], *args, **kwargs): ...

    def process(self, data, *args, **kwargs):
        if isinstance(data, str):
            if self._mode == 'rm':
                if self._pos_type == 'prefix':
                    return data[len(self._label):] if data.startswith(self._label) else data
                else:
                    return data[:len(self._label)] if data.endswith(self._label) else data
            else:
                if self._pos_type == 'prefix':
                    return self._label + self._label_glue + data
                else:
                    return data + self._label_glue + self._label
        else:
            if self._mode == 'rm':
                if self._pos_type == 'prefix':
                    return [el[len(self._label):] if el.startswith(self._label) else el for el in data]
                else:
                    return [el[:len(self._label)] if el.endswith(self._label) else el for el in data]
            else:
                if self._pos_type == 'prefix':
                    return [self._label + self._label_glue + el for el in data]
                else:
                    return [el + self._label_glue + self._label for el in data]

    @property
    def label(self):
        return self._label
