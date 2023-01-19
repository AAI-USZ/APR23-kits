from typing import overload, List, Union

import requests
from requests import status_codes

from aprkits.types import CommandSequencerOpts, CommandSequenceOneOutput, CommandSequenceManyOutput, \
    CommandSequenceBatchOutput

_HOST = 'localhost'
_PROXIES = {
    'http': f'http://{_HOST}:8080',
    'https': f'http://{_HOST}:8080'
}
_BASE_URL = f'http://{_HOST}/api/command/sequence'


@overload
def get_command_sequence(
        a: List[str], b: List[str], *, opts: Union[dict, CommandSequencerOpts] = None
) -> CommandSequenceOneOutput: ...


@overload
def get_command_sequence(
        a: List[List[str]], b: List[List[str]], *, opts: Union[dict, CommandSequencerOpts] = None
) -> CommandSequenceManyOutput: ...


@overload
def get_command_sequence(
        a: List[List[str]], b: List[List[str]], *, batch_size: int, opts: Union[dict, CommandSequencerOpts] = None
) -> CommandSequenceBatchOutput: ...


def get_command_sequence(a, b, *, batch_size: int = None, opts: Union[dict, CommandSequencerOpts] = None):
    min_len = min(len(a), len(b))
    if min_len <= 0:
        return ''
    if len(a) != len(b):
        a = a[:min_len]
        b = b[:min_len]

    req = {
        'a': a,
        'b': b,
        'batchSize': batch_size,
        'opts': opts
    }

    if isinstance(a[0], str):
        resp = requests.post(
            _BASE_URL + '/one',
            json=req,
            proxies=_PROXIES
        )
        if resp.status_code == status_codes.codes['ok']:
            json_obj = resp.json()
            return CommandSequenceOneOutput(
                command=json_obj['command'],
                labels=json_obj['labels'],
                numbers=json_obj['numbers']
            )
    elif isinstance(a[0], (list, tuple)) and batch_size is None:
        resp = requests.post(
            _BASE_URL + '/many',
            json=req,
            proxies=_PROXIES
        )
        if resp.status_code == status_codes.codes['ok']:
            json_obj = resp.json()
            return CommandSequenceManyOutput(
                command=json_obj['command'],
                labels=json_obj['labels'],
                numbers=json_obj['numbers']
            )
    else:
        resp = requests.post(
            _BASE_URL + '/batch',
            json=req,
            proxies=_PROXIES
        )
        if resp.status_code == status_codes.codes['ok']:
            json_obj = resp.json()
            return CommandSequenceBatchOutput(
                command=json_obj['command'],
                labels=json_obj['labels'],
                numbers=json_obj['numbers']
            )

    raise TypeError(f'Can not return {None}')
