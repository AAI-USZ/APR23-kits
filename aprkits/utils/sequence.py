import re
from typing import List, Tuple, Optional, overload, Union


def _delete(cmd: str, *, i1: int, i2: int, delete_token: str):
    return cmd + f'{delete_token}({i1}:{i2}) '


def _insert(
        cmd: str,
        *,
        i1: int,
        j1: int,
        j2: int,
        data: List[str],
        insert_token: str
):
    return cmd + f'{insert_token}({i1}) {" ".join(data[j1:j2])} '


def _posparse(posstr: str):
    pos = posstr.strip('()').split(':')
    if len(pos) < 2:
        return int(pos[0]), None
    return int(pos[0]), int(pos[1])


def _swapreplacements(parsedcmd: List[Tuple[str, int, Optional[int], List[str]]]):
    _dummy = ('', -1, -1, [])
    swapped = [
        _prev if _curr[0] == 'insert' and _prev[0] == 'delete' and _curr[1] == _prev[1]
        else _next if _curr[0] == 'delete' and _next[0] == 'insert' and _curr[1] == _next[1]
        else _curr
        for _prev, _curr, _next in zip(
            [_dummy] + parsedcmd[:-1],
            parsedcmd,
            parsedcmd[1:] + [_dummy]
        )
    ]
    return swapped


def get_command_sequence(
        op_codes: List[Tuple[str, int, int, int, int]],
        b_tokens: List[str],
        delete_token: str = '</[DELETE]/>',
        insert_token: str = '</[INSERT]/>'
):
    command_sequence = ''

    for op, i1, i2, j1, j2 in op_codes:
        if op == 'delete':
            command_sequence = _delete(command_sequence, i1=i1, i2=i2, delete_token=delete_token)
        elif op == 'insert':
            command_sequence = _insert(command_sequence, i1=i1, j1=j1, j2=j2, data=b_tokens, insert_token=insert_token)
        elif op == 'replace':
            command_sequence = _insert(_delete(
                command_sequence, i1=i1, i2=i2, delete_token=delete_token),
                i1=i1, j1=j1, j2=j2, data=b_tokens, insert_token=insert_token)

    return command_sequence.strip()


def parse_command_sequence(
        command_sequence: str, delete_token: str = '</[DELETE]/>', insert_token: str = '</[INSERT]/>'
) -> List[Tuple[str, int, Optional[int], List[str]]]:
    commands = re.findall(
        rf'{re.escape(delete_token)}\(\d+:\d+\)|'
        rf'{re.escape(insert_token)}.*?(?={re.escape(insert_token)}|{re.escape(delete_token)}|$)',
        command_sequence)
    commands = [
        re.findall(
            rf'{re.escape(delete_token)}\(\d+:\d+\)|'
            rf'{re.escape(insert_token)}\(\d+\)|\S+',
            cmd)
        for cmd in commands]
    commands = [
        (
            re.search(r'(?<=\[)\S+(?=])', cmd[0]).group().lower(),
            *_posparse(re.search(r'\(.*\)', cmd[0]).group()),
            cmd[1:]
        )
        for cmd in commands
    ]
    return commands


def transform_by_command_sequence(
        command_sequence: str,
        a_tokens: List[str],
        delete_token: str = '</[DELETE]/>',
        insert_token: str = '</[INSERT]/>'
):
    parsed_cmd = parse_command_sequence(command_sequence, delete_token=delete_token, insert_token=insert_token)

    tokens = a_tokens
    for op, i1, i2, data in _swapreplacements(parsed_cmd)[::-1]:
        if op == 'insert':
            tokens = tokens[:i1] + data + tokens[i1:]
        elif op == 'delete':
            tokens = tokens[:i1] + tokens[i2:]

    return tokens


def _extract_numberings_impl(
        command_sequence: Union[str, List[str]], delete_token: str = '</[DELETE]/>', insert_token: str = '</[INSERT]/>'
):
    numberings = re.findall(
        rf'(?<={re.escape(delete_token)})\(\d+:\d+\)|(?<={re.escape(insert_token)})\(\d+\)', command_sequence)
    numberings = ' '.join([' '.join(num.strip('()').split(':')) for num in numberings])
    return numberings


@overload
def extract_numberings(
        command_sequence: str, delete_token: str = '</[DELETE]/>', insert_token: str = '</[INSERT]/>'
) -> str: ...


@overload
def extract_numberings(
        command_sequence: List[str], delete_token: str = '</[DELETE]/>', insert_token: str = '</[INSERT]/>'
) -> List[str]: ...


def extract_numberings(
        command_sequence: Union[str, List[str]], delete_token: str = '</[DELETE]/>', insert_token: str = '</[INSERT]/>'
):
    if isinstance(command_sequence, list):
        return [
            _extract_numberings_impl(cmd_seq, delete_token=delete_token, insert_token=insert_token)
            for cmd_seq in command_sequence
        ]
    return _extract_numberings_impl(command_sequence, delete_token=delete_token, insert_token=insert_token)


def _extract_labels_impl(
        command_sequence: str,
        delete_token: str = '</[DELETE]/>',
        insert_token: str = '</[INSERT]/>',
        location_token: str = '</[LOC]/>'
):
    labels = re.sub(
        rf'(?<={re.escape(delete_token)})\(\d+:\d+\)',
        f' {location_token}{location_token}'.strip(),
        command_sequence
    )
    labels = re.sub(
        rf'(?<={re.escape(insert_token)})\(\d+\)',
        f' {location_token}'.strip(),
        labels
    )
    return labels


@overload
def extract_labels(
        command_sequence: str,
        delete_token: str = '</[DELETE]/>',
        insert_token: str = '</[INSERT]/>',
        location_token: str = '</[LOC]/>'
): ...


@overload
def extract_labels(
        command_sequence: List[str],
        delete_token: str = '</[DELETE]/>',
        insert_token: str = '</[INSERT]/>',
        location_token: str = '</[LOC]/>'
): ...


def extract_labels(
        command_sequence: str,
        delete_token: str = '</[DELETE]/>',
        insert_token: str = '</[INSERT]/>',
        location_token: str = '</[LOC]/>'
):
    if isinstance(command_sequence, list):
        return [
            _extract_labels_impl(
                cmd_seq, delete_token=delete_token, insert_token=insert_token, location_token=location_token
            )
            for cmd_seq in command_sequence
        ]
    return _extract_labels_impl(
        command_sequence, delete_token=delete_token, insert_token=insert_token, location_token=location_token
    )
