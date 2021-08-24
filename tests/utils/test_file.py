import shutil
from pathlib import Path
from typing import Generator, Optional, List, Callable

import pytest

from tests.utils.test_utils import prepare_temporary_dir
from utils.file import dump_text_file, get_directory_content


@pytest.fixture
def mock_directory_path() -> Generator[str, None, None]:
    output_path = prepare_temporary_dir(directory_name="mock_directory_path")
    yield output_path
    shutil.rmtree(output_path)


def mock_directory_content(directory_path: str) -> None:
    dump_text_file(Path(directory_path).joinpath('file_1.json').as_posix(), '')
    dump_text_file(Path(directory_path).joinpath('file_2.txt').as_posix(), '')
    dump_text_file(Path(directory_path).joinpath('file_3.txt').as_posix(), '')


@pytest.mark.parametrize(
    "extension, mock_callback, expected_result",
    [
        (
            None,
            lambda x: None,
            0
        ),  # empty directory
        (
            None,
            mock_directory_content,
            3
        ),  # directory contain 3 files
        (
            'json',
            mock_directory_content,
            1
        ),  # directory contain 1 .json file
        (
            'txt',
            mock_directory_content,
            2
        ),  # directory contain 2 .txt files
        (
            'avi',
            mock_directory_content,
            0
        ),  # directory contain 0 .avi files
    ]
)
def test_get_directory_content(
    mock_directory_path: str,
    extension: Optional[str],
    mock_callback: Callable[[str], None],
    expected_result: List[str]
) -> None:
    mock_callback(mock_directory_path)
    result = get_directory_content(directory_path=mock_directory_path, extension=extension)
    assert len(result) == expected_result
