from models import common
import pytest

def test_autopad():
    k = 1
    p = 2
    autopad = common.autopad(k, p)