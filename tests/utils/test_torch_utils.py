from utils import torch_utils
import pytest


class TestTorchUtils:

    def test_select_device(self):
        device = 'cpu'
        batch_size = 0
        torch_utils.select_device(device, batch_size)