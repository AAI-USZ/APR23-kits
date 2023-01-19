from typing import List, Optional, Union, overload

import requests
from requests import status_codes

from aprkits.types import FlatNode, GrapherOptions

_HOST = 'localhost'
_PROXIES = {
    'http': f'http://{_HOST}:8080',
    'https': f'http://{_HOST}:8080'
}
_BASE_URL = f'http://{_HOST}/api/tree-sitter/grapher'


@overload
def get_flat_graph(
        code: str, *, opts: Union[dict, GrapherOptions]
) -> Optional[List[FlatNode]]: ...


@overload
def get_flat_graph(
        codes: List[str], *, opts: Union[dict, GrapherOptions] = None
) -> Optional[List[List[FlatNode]]]: ...


@overload
def get_flat_graph(
        codes: List[str], *, batch_size: int, opts: Union[dict, GrapherOptions] = None
) -> Optional[List[List[List[FlatNode]]]]: ...


def get_flat_graph(code_or_codes, *, batch_size=None, opts=None):
    if isinstance(opts, GrapherOptions):
        opts = {
            'lang': opts.lang,
            'flattenCode': opts.flatten_code
        }
    try:
        chunk = opts.get('chunk_size') or opts.chunk_size
    except AttributeError:
        chunk = GrapherOptions().chunk_size

    if isinstance(code_or_codes, str):
        resp = requests.post(
            _BASE_URL + '/one',
            json={
                'code': code_or_codes,
                'opts': opts
            },
            proxies=_PROXIES
        )

        if resp.status_code == status_codes.codes['ok']:
            json_obj = resp.json()
            return json_obj

    elif isinstance(code_or_codes, (list, tuple)):
        res = []

        for chunk_idx in range(len(code_or_codes) // chunk + 1):
            code_chunk = code_or_codes[chunk_idx * chunk:chunk_idx * chunk + chunk]
            resp = requests.post(
                _BASE_URL + '/many',
                json={
                    'codes': code_chunk,
                    'opts': opts
                },
                proxies=_PROXIES
            )

            if resp.status_code == status_codes.codes['ok']:
                json_obj = resp.json()
                res += json_obj
            else:
                return None

        return res

    else:
        res = []

        for chunk_idx in range(len(code_or_codes) // chunk):
            code_chunk = code_or_codes[chunk_idx * chunk:chunk_idx * chunk + chunk]
            resp = requests.post(
                _BASE_URL + '/batch',
                json={
                    'codes': code_chunk,
                    'batchSize': batch_size,
                    'opts': opts
                },
                proxies=_PROXIES
            )

            if resp.status_code == status_codes.codes['ok']:
                json_obj = resp.json()
                res += json_obj
            else:
                return None

        return res

    return None
