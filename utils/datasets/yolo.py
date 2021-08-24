import os
import glob
from pathlib import Path

from typing import List, Union

from utils.datasets_old import IMG_FORMATS
from utils.file import read_text_file_lines


def load_image_names_from_paths(paths: Union[str, List[str]]) -> List[str]:
    image_paths = []
    for path in paths if isinstance(paths, list) else [paths]:
        path = Path(path)  # os-agnostic
        if path.is_dir():  # dir
            image_paths += glob.glob(str(path / '**' / '*.*'), recursive=True)
        elif path.is_file():  # file
            local_paths = read_text_file_lines(path)
            parent = str(path.parent) + os.sep
            image_paths += [
                local_path.replace('./', parent) if local_path.startswith('./') else local_path
                for local_path
                in local_paths
            ]
        else:
            raise Exception(f'{path} does not exist')
    return sorted([x.replace('/', os.sep) for x in image_paths if x.split('.')[-1].lower() in IMG_FORMATS])
