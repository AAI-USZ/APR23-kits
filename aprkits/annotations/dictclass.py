def _dict_getitem(self, item):
    return self.__dict__[item]


def _dict_contains(self, item):
    return item in self.__dict__


def _process_class(cls=None):
    setattr(cls, '__getitem__', _dict_getitem)
    setattr(cls, '__contains__', _dict_contains)
    return cls


def dictclass(cls=None):
    def wrap(cls):
        return _process_class(cls)

    if cls is None:
        return wrap

    return wrap(cls)
