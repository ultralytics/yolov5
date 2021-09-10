import unittest
import os
import shutil
from utils.general import check_dataset

class TestCheckDatasets(unittest.TestCase):

    def test_download_from_yaml(self) -> None:
        directory = '../datasets/coco128'
        file_path = '../datasets/coco128.zip'

        self._cleanup(directory, file_path)

        check_dataset('data/coco128.yaml')
        result = self._folder_exists(directory)
        assert result is True

    def test_download_from_yaml_to_custom(self) -> None:
        directory = 'testing_check_dataset/coco128'
        file_path = 'testing_check_dataset/coco128.zip'

        self._cleanup(directory, file_path)
        check_dataset('data/coco128.yaml', save_root='testing_check_dataset')
        result = self._folder_exists(directory)
        assert result is True

    def test_download_from_url(self):

        url = "https://firebasestorage.googleapis.com/v0/b/ultralytics-hub.appspot.com/o/users%2Fx6Ic1iRCVHUrT0GnPXX6TinOIAJ2%2Fdatasets%2F93iomucdcWyn9althV2A%2Fcoco128.zip?alt=media&token=d334fec0-577f-49e3-8fcc-947c0b791c67"

        directory = '../datasets/coco128'
        file_path = '../datasets/coco128.zip'

        self._cleanup(directory, file_path)

        check_dataset(url)

        result = self._folder_exists(directory)
        assert result is True

    def test_download_from_url_to_custom(self):
        url = "https://firebasestorage.googleapis.com/v0/b/ultralytics-hub.appspot.com/o/users%2Fx6Ic1iRCVHUrT0GnPXX6TinOIAJ2%2Fdatasets%2F93iomucdcWyn9althV2A%2Fcoco128.zip?alt=media&token=d334fec0-577f-49e3-8fcc-947c0b791c67"

        directory = 'testing_check_dataset/coco128'
        file_path = 'testing_check_dataset/coco128.zip'

        self._cleanup(directory, file_path)

        check_dataset(url, save_root='testing_check_dataset')

        result = self._folder_exists(directory)
        assert result is True

    def _file_exists(self, path):
        return os.path.isfile(path)
    
    def _folder_exists(self, path):
        return os.path.isdir(path)

    def _remove_directory(self, path):
        if self._folder_exists(path):
            shutil.rmtree(path)

    def _remove_file(self, path):
        if self._file_exists(path):
            os.remove(path)

    def _cleanup(self, directory, file_path):
        self._remove_directory(directory)
        self._remove_file(file_path)
