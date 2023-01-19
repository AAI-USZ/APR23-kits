import shutil
from os import PathLike
from pathlib import Path
from typing import Union


def transform_suffix(
        path_to_dir: Union[str, PathLike],
        output_path: Union[str, PathLike],
        new_suffix: str
):
    path_to_dir = Path(path_to_dir)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    for file in path_to_dir.iterdir():
        if file.is_file():
            shutil.copy(file, Path(output_path, file.stem + new_suffix))
