from models import experimental
import pytest

def test_crossconv():
    c1 = 1
    c2 = 1
    k = 1
    s = 1
    g = 1
    e = 1
    shortcut = False
    experimental.CrossConv(c1, c2, k, s, g, e, shortcut)