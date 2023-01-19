from collections import Iterable


def json_dict_to_md(json_dict: dict):
    lvlines = []

    def _iter(o, level=0):
        if isinstance(o, dict):
            for k, v in o.items():
                lvlines.append([level, f'{" ".join([s.title() for s in str(k).split("_")]).strip()}:'])
                _iter(v, level + 1)
        elif isinstance(o, (str, int, float, bool)) or o is None:
            lvlines[-1][1] += f' {o}'
        elif isinstance(o, Iterable):
            for i in o:
                lvlines.append([level, f'{i}:'])
                _iter(i, level + 1)

    _iter(json_dict)

    lines = [' '.join((f'{"#" * (x + 1)} {"&nbsp; " * x * 2}', val)) for x, val in lvlines]
    md = '\n'.join(lines)
    return md
