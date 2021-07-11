import os
from typing import Union, Tuple, List


def list_files_with_extension(root_path: str, extensions: Union[str, Tuple[str, ...]]) -> List[str]:
    return [file for file in os.listdir(root_path) if file.endswith(extensions)]
