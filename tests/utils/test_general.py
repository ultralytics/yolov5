from utils import general
import pytest

class TestGeneral:
    def test_set_logging(self):
        rank = -1
        general.set_logging(rank)

    def test_init_seeds(self):
        seed = 0
        general.init_seeds(seed)
    
    def test_get_latest_run(self):
        search_dir = "test"
        general.get_latest_run(search_dir)
    
    def test_check_git_status(self):
        general.check_git_status()
    
    def test_check_img_size(self):
        img_size = 32
        s = 32
        general.check_img_size(img_size, s)

    @pytest.mark.xfail(raises=AssertionError)
    def test_check_file(self):
        file = "test.txt"
        general.check_file(file)

    @pytest.mark.xfail(raises=AttributeError)    
    def test_check_dataset(self):
        dict = []
        general.check_dataset(dict)
    
    def test_make_divisible(self):
        x = 1
        divisor = 2
        general.make_divisible(x, divisor)