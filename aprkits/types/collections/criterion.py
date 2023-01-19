from collections import UserDict
from typing import TypedDict, Optional, Union

# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss


NamedLoss = Union[
    TypedDict(
        'NamedLoss',
        {
            'name': str,
            'loss': _Loss,
            'weight': Optional[float]
        }
    ),
    TypedDict(
        'NamedLoss',
        {
            'name': str,
            'loss': _Loss
        }
    )
]


class _WeightedLoss(UserDict):
    def __init__(self, wl):
        super().__init__(wl)

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(f'\'{self.__class__.__name__}\' object has no attribute \'{item}\'')
        return self.data[item]


class CriterionCollection(UserDict):
    def __init__(self, *criterions: NamedLoss):
        super().__init__({
            el['name']: _WeightedLoss({
                'loss': el['loss'],
                'weight': el['weight'] if 'weight' in el and el['weight'] is not None else 1.
            })
            for el in criterions
        })

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(f'\'{self.__class__.__name__}\' object has no attribute \'{item}\'')
        return self.data[item]
