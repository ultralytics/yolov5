from utils import loss
import pytest
import mock

class TestLoss:

    @pytest.mark.xfail(raises=RuntimeError)    
    def test_compute_loss(self):
        p = 1
        targets = mock.Mock()
        targets.device = 'test'
        model = 3
        tuple = loss.compute_loss(p, targets, model)