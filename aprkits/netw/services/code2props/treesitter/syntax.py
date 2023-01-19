import requests
from requests import status_codes

_HOST = 'localhost'
_PROXIES = {
    'http': f'http://{_HOST}:8080',
    'https': f'http://{_HOST}:8080'
}
_BASE_URL = f'http://{_HOST}/api/tree-sitter/syntax'


def get_treesitter_syntax(*, lang: str = None, code: str = None):
    resp = requests.get(
        _BASE_URL,
        params={
            'lang': lang,
            'code': code,
        },
        proxies=_PROXIES
    )
    if resp.status_code == status_codes.codes['ok']:
        json_obj = resp.json()
        return json_obj
    return None
