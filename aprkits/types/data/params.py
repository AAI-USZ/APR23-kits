from pathlib import Path
from typing import TypedDict


ModelParams = TypedDict('ModelParams', {'path': Path, 'params': dict})
