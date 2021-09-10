import unittest
from pathlib import Path
from utils.general import has_extension

class TestCheckSuffix(unittest.TestCase):

    def test_string_is_extension(self) -> bool:
        file = 'test.txt'
        extension = '.txt'
        result = has_extension(file, extension)
        assert result is True

    def test_string_is_not_extension(self) -> bool:
        file = 'test.txt'
        extension = '.fail'
        result = has_extension(file, extension)
        assert result is False

    def test_string_is_one_extension(self) -> bool:
        file = 'test.txt'
        extension = ('.txt', '.exe')
        result = has_extension(file, extension)
        assert result is True

    def test_string_is_not_one_extension(self) -> bool:
        file = 'test.txt'
        extension = ('.fail', '.exe')
        result = has_extension(file, extension)
        assert result is False

    def test_path_is_extension(self) -> bool:
        file = Path('test.txt')
        extension = '.txt'
        result = has_extension(file, extension)
        assert result is True

    def test_path_is_not_extension(self) -> bool:
        file = Path('test.txt')
        extension = '.fail'
        result = has_extension(file, extension)
        assert result is False

    def test_path_is_one_extension(self) -> bool:
        file = Path('test.txt')
        extension = ('.txt', '.exe')
        result = has_extension(file, extension)
        assert result == True

    def test_path_is_not_one_extension(self) -> bool:
        file = Path('test.txt')
        extension = ('.fail', '.exe')
        result = has_extension(file, extension)
        assert result == False