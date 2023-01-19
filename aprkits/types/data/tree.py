from typing import List, Optional, TypedDict


Node = TypedDict(
    'Node',
    {
        'id': int,
        'type': str,
        'name': str,
        'value': str,
        'parent': Optional[int],
        'children': List[int],
        'coeff': float
    }
)

TreePathData = TypedDict(
    'TreePathData',
    {
        'tokens': List[str],
        'nodes': List[Node],
        'treePaths': List[List[Node]]
    }
)

FlatNode = TypedDict(
    'FlatNode',
    {
        'type': str,
        'startNode': str,
        'endNode': str,
        'childCount': int
    }
)
