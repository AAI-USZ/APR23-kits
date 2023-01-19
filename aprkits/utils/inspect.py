import functools


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj_, attr_):
        return getattr(obj_, attr_, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
