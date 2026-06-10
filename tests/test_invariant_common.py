import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.common import _validate_ssrf_url

BLOCKED_URLS = [
    "http://169.254.169.254/latest/meta-data/",   # AWS metadata / link-local
    "http://192.168.1.1/admin",                    # RFC1918 private
    "http://127.0.0.1/internal",                   # Loopback by IP
    "http://localhost/internal",                   # Loopback by name
]

ALLOWED_URLS = [
    "https://ultralytics.com/images/zidane.jpg",   # Valid public URL
]


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_ssrf_blocked_urls(url):
    """Validator must raise ValueError for internal/private targets."""
    with pytest.raises(ValueError):
        _validate_ssrf_url(url)


@pytest.mark.parametrize("url", ALLOWED_URLS)
@pytest.mark.network
def test_ssrf_allowed_urls(url):
    """Validator must not raise for legitimate public URLs."""
    _validate_ssrf_url(url)
