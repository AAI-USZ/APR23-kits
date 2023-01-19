from typing import TypedDict

from .data import DataConfig


Config = TypedDict('Config', {
    'data': DataConfig
})
