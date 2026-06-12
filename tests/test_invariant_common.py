# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.common import _request_ssrf_url, _validate_ssrf_url

BLOCKED_URLS = [
    "http://169.254.169.254/latest/meta-data/",  # AWS metadata / link-local
    "http://192.168.1.1/admin",  # RFC1918 private
    "http://127.0.0.1/internal",  # Loopback by IP
    "http://localhost/internal",  # Loopback by name
]

ALLOWED_URLS = [
    "https://ultralytics.com/images/zidane.jpg",  # Valid public URL
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


def test_ssrf_redirect_target_is_validated(monkeypatch):
    """Redirects must not bypass URL validation."""
    calls = []

    class RedirectResponse:
        is_redirect = True
        headers = {"location": "http://169.254.169.254/latest/meta-data/"}
        url = "https://example.com/image.jpg"

        def close(self):
            pass

    class FakeSession:
        def get(self, url, **kwargs):
            assert kwargs["allow_redirects"] is False
            return RedirectResponse()

    def fake_validate(url):
        calls.append(url)
        if "169.254.169.254" in url:
            raise ValueError("blocked")

    monkeypatch.setattr("models.common._validate_ssrf_url", fake_validate)
    monkeypatch.setattr("models.common.requests.Session", FakeSession)

    with pytest.raises(ValueError):
        _request_ssrf_url("https://example.com/image.jpg")
    assert calls == ["https://example.com/image.jpg", "http://169.254.169.254/latest/meta-data/"]
