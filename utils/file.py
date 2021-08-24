from glob import glob
from pathlib import Path
from typing import List, Union, Optional


def read_text_file_lines(file_path: Union[str, Path], remove_blank: bool = True) -> List[str]:
    with open(file_path, "r") as file:
        lines = [l.strip(' \n') for l in file.readlines()]
        if remove_blank:
            return list(filter(lambda l: len(l) > 0, lines))
        else:
            return lines


def get_directory_content(directory_path: str, extension: Optional[str] = None) -> List[str]:
    wild_card = '*' if extension is None else f'*.{extension}'
    pattern = Path(directory_path).joinpath(wild_card).as_posix()
    return glob(pattern)


def dump_text_file(file_path: str, content: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
