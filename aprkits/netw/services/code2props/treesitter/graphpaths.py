from typing import List, Optional

import requests
from requests import status_codes

from aprkits.types import TreeSitterPathOptions, TreePathData


_HOST = 'localhost'
_PROXIES = {
    'http': f'http://{_HOST}:8080',
    'https': f'http://{_HOST}:8080'
}
_BASE_URL = f'http://{_HOST}/api/tree-sitter/paths'


def get_paths_one(
        code: str,
        lang: str,
        strict: bool = None,
        flatten_code: bool = None,
        options: TreeSitterPathOptions = None
) -> Optional[TreePathData]:
    if options is None:
        options = {
            'additionalLeafNodes': None,
            'regexSignal': None,
            'getChildCoeffs': None
        }
    else:
        options = {
            'additionalLeafNodes': options.additional_leaf_nodes,
            'regexSignal': options.regex_signal,
            'getChildCoeffs': options.get_child_coeffs
        }

    resp = requests.post(
        _BASE_URL + '/one',
        json={
            'code': code,
            'lang': lang,
            'strict': strict,
            'flattenCode': flatten_code,
            'options': options
        },
        proxies=_PROXIES
    )

    if resp.status_code == status_codes.codes['ok']:
        json_obj: TreePathData = resp.json()
        # making treePaths part so that it references nodes -> essentially they will be the same object
        json_obj['treePaths'] = [
            [json_obj['nodes'][node['id']] for node in path]
            for path in json_obj['treePaths']
        ]
        return json_obj

    return None


def get_paths_many(
        codes: List[str],
        lang: str,
        flatten_code: bool = None,
        options: TreeSitterPathOptions = None
) -> Optional[List[TreePathData]]:
    if options is None:
        options = {
            'additionalLeafNodes': None,
            'regexSignal': None,
            'getChildCoeffs': None
        }
    else:
        options = {
            'additionalLeafNodes': options.additional_leaf_nodes,
            'regexSignal': options.regex_signal,
            'getChildCoeffs': options.get_child_coeffs
        }

    resp = requests.post(
        _BASE_URL + '/many',
        json={
            'codes': codes,
            'lang': lang,
            'flattenCode': flatten_code,
            'options': options
        },
        proxies=_PROXIES
    )

    if resp.status_code == status_codes.codes['ok']:
        json_obj: List[TreePathData] = resp.json()
        # making treePaths part so that it references nodes -> essentially they will be the same object
        json_obj = [
            TreePathData(
                tokens=treePathData['tokens'],
                nodes=treePathData['nodes'],
                treePaths=[
                    [treePathData['nodes'][node['id']] for node in path]
                    for path in treePathData['treePaths']
                ]
            )
            for treePathData in json_obj
        ]
        return json_obj

    return None


def get_paths_batch(
        codes: List[str],
        lang: str,
        batch_size: int,
        flatten_code: bool = None,
        options: TreeSitterPathOptions = None
) -> Optional[List[List[TreePathData]]]:
    if options is None:
        options = {
            'additionalLeafNodes': None,
            'regexSignal': None,
            'getChildCoeffs': None
        }
    else:
        options = {
            'additionalLeafNodes': options.additional_leaf_nodes,
            'regexSignal': options.regex_signal,
            'getChildCoeffs': options.get_child_coeffs
        }

    resp = requests.post(
        _BASE_URL + '/batch',
        json={
            'codes': codes,
            'lang': lang,
            'batchSize': batch_size,
            'flattenCode': flatten_code,
            'options': options
        },
        proxies=_PROXIES
    )

    if resp.status_code == status_codes.codes['ok']:
        json_obj: List[List[TreePathData]] = resp.json()
        # making treePaths part so that it references nodes -> essentially they will be the same object
        json_obj = [
            [
                TreePathData(
                    tokens=treePathData['tokens'],
                    nodes=treePathData['nodes'],
                    treePaths=[
                        [treePathData['nodes'][node['id']] for node in path]
                        for path in treePathData['treePaths']
                    ]
                )
                for treePathData in batch
            ]
            for batch in json_obj
        ]
        return json_obj

    return None
