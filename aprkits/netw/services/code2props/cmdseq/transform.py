from typing import overload, List, Union

import requests
from requests import status_codes

from aprkits.types import CommandSequencerOpts

_HOST = 'localhost'
_PROXIES = {
    'http': f'http://{_HOST}:8080',
    'https': f'http://{_HOST}:8080'
}
_BASE_URL = f'http://{_HOST}/api/command/transform'


@overload
def transform_by_command(
        cmd_seq: str,
        a: List[str],
        opts: Union[dict, CommandSequencerOpts] = None
) -> List[str]: ...


@overload
def transform_by_command(
        cmd_seq: List[str],
        a: List[List[str]],
        opts: Union[dict, CommandSequencerOpts] = None
) -> List[List[str]]: ...


@overload
def transform_by_command(
        cmd_seq: List[str],
        a: List[List[str]],
        batch_size: int,
        opts: Union[dict, CommandSequencerOpts] = None
) -> List[List[List[str]]]: ...


def transform_by_command(cmd_seq, a, batch_size=None, opts: Union[dict, CommandSequencerOpts] = None):
    if isinstance(cmd_seq, (list, tuple)):
        min_len = min(len(cmd_seq), len(a))
        if min_len <= 0:
            return []
        if len(cmd_seq) != len(a):
            cmd_seq = cmd_seq[:min_len]
            a = a[:min_len]

    req = {
        'cmd': cmd_seq,
        'a': a,
        'batchSize': batch_size,
        'opts': opts
    }

    if isinstance(cmd_seq, str):
        resp = requests.post(
            _BASE_URL + '/one',
            json=req,
            proxies=_PROXIES
        )
    elif isinstance(cmd_seq, (list, tuple)) and batch_size is None:
        resp = requests.post(
            _BASE_URL + '/many',
            json=req,
            proxies=_PROXIES
        )
    else:
        resp = requests.post(
            _BASE_URL + '/batch',
            json=req,
            proxies=_PROXIES
        )

    if resp.status_code == status_codes.codes['ok']:
        json_obj = resp.json()
        return json_obj

    raise TypeError(f'Can not return {None}')
