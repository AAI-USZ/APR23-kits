from abc import ABC, abstractmethod


class _Processor(ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, data, *args, **kwargs): ...


class _StringProcessor(_Processor):
    def __init__(self, clean_spaces: bool = True):
        super().__init__()
        self._clean_spaces = clean_spaces

    def __call__(self, *args, **kwargs):
        out = self.process(*args, **kwargs)
        if isinstance(out, str):
            if self._clean_spaces:
                return out.strip()
        if self._clean_spaces:
            return [el.strip() for el in out]
        return out

    @abstractmethod
    def process(self, data, *args, **kwargs): ...
