# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.common import DetectMultiBackend, _request_ssrf_url, _validate_ssrf_url
from utils.segment.general import scale_image

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


def test_scale_image_many_masks():
    """scale_image must resize more than 512 masks, which OpenCV cannot resize as channels directly."""
    masks = np.arange(4 * 5 * 513, dtype=np.float32).reshape(4, 5, 513)

    resized = scale_image((4, 5), masks, (8, 10, 3))
    expected = np.stack([cv2.resize(masks[:, :, i], (10, 8)) for i in range(masks.shape[2])], axis=2)

    assert resized.shape == (8, 10, 513)
    np.testing.assert_allclose(resized, expected)


def test_tensorrt_deserialize_failure_reports_environment(monkeypatch, tmp_path):
    """TensorRT engine incompatibility must fail with an actionable runtime error."""
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"bad-engine")

    class FakeLogger:
        INFO = 1

        def __init__(self, *_args):
            pass

    class FakeRuntime:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def deserialize_cuda_engine(self, data):
            assert data == b"bad-engine"
            return None

    fake_trt = SimpleNamespace(__version__="8.6.1", Logger=FakeLogger, Runtime=FakeRuntime)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    with pytest.raises(RuntimeError, match="same TensorRT version"):
        DetectMultiBackend(str(engine))
