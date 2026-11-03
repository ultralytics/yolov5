# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.common import DetectMultiBackend, _request_ssrf_url, _validate_ssrf_url
import torch
from utils.metrics import ConfusionMatrix
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

    fake_trt = SimpleNamespace(__version__="8.6.1", Logger=FakeLogger, Runtime=FakeRuntime)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    with pytest.raises(RuntimeError, match="same TensorRT version"):
        DetectMultiBackend(str(engine))


def test_confusion_matrix_none_detections_2d_labels():
    """Verify ConfusionMatrix.process_batch handles documented 2D (M, 5) labels when detections=None."""
    nc = 80
    cm = ConfusionMatrix(nc=nc)

    labels = torch.tensor(
        [
            [0.0, 100.0, 200.0, 150.0, 250.0],
            [5.0, 10.0, 20.0, 30.0, 40.0],
        ]
    )

    cm.process_batch(detections=None, labels=labels)

    assert cm.matrix[nc, 0] == 1.0, f"Expected FN count 1 for class 0, got {cm.matrix[nc, 0]}"
    assert cm.matrix[nc, 5] == 1.0, f"Expected FN count 1 for class 5, got {cm.matrix[nc, 5]}"
    assert cm.matrix.sum() == 2.0, f"Expected total matrix sum 2.0, got {cm.matrix.sum()}"

    for corrupted_col in [10, 20, 30, 40]:
        assert cm.matrix[nc, corrupted_col] == 0.0, (
            f"Coordinate {corrupted_col} was mistakenly counted as a false negative!"
        )


def test_confusion_matrix_none_detections_1d_labels():
    """Verify backward compatibility with 1D labels slice as passed from val.py."""
    nc = 80
    cm = ConfusionMatrix(nc=nc)

    labels_1d = torch.tensor([0, 5])
    cm.process_batch(detections=None, labels=labels_1d)

    assert cm.matrix[nc, 0] == 1.0
    assert cm.matrix[nc, 5] == 1.0
    assert cm.matrix.sum() == 2.0


def test_confusion_matrix_none_detections_empty_labels():
    """Verify handling of empty labels when detections=None."""
    nc = 80
    cm = ConfusionMatrix(nc=nc)

    labels_empty = torch.zeros((0, 5))
    cm.process_batch(detections=None, labels=labels_empty)
    assert cm.matrix.sum() == 0.0
