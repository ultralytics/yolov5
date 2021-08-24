from pathlib import Path
from typing import List, Union


def read_text_file_lines(file_path: Union[str, Path], remove_blank: bool = True) -> List[str]:
    with open(file_path, "r") as file:
        lines = [l.strip(' \n') for l in file.readlines()]
        if remove_blank:
            return list(filter(lambda l: len(l) > 0, lines))
        else:
            return lines
