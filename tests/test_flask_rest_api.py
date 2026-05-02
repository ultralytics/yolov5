# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
import io

import pytest
from PIL import Image

from utils.flask_rest_api.restapi import DETECTION_URL, MAX_IMAGE_SIZE, app, models

MODEL_NAME = "yolov5s"
DETECTION_PATH = DETECTION_URL.replace("<model>", MODEL_NAME)


class DummyResults:
    class _Pandas:
        xyxy = [type("DummyDf", (), {"to_json": lambda self, orient: "[]"})()]

    def pandas(self):
        return self._Pandas()


class DummyModel:
    def __call__(self, im, size=640):
        return DummyResults()


@pytest.fixture(autouse=True)
def setup_model():
    models.clear()
    models[MODEL_NAME] = DummyModel()
    yield
    models.clear()


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    return app.test_client()


def make_image_bytes(fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buf, format=fmt)
    buf.seek(0)
    return buf


def post_image(client, file_obj, filename):
    return client.post(
        DETECTION_PATH,
        data={"image": (file_obj, filename)},
        content_type="multipart/form-data",
    )


def test_rejects_non_image_upload_with_allowed_extension(client):
    response = post_image(client, io.BytesIO(b"not really an image"), "fake.jpg")
    assert response.status_code == 400
    assert b"Invalid image file" in response.data


def test_rejects_upload_with_disallowed_extension(client):
    response = post_image(client, io.BytesIO(b"hello"), "payload.txt")
    assert response.status_code == 400
    assert b"Invalid file type" in response.data


def test_rejects_oversized_upload(client):
    response = post_image(client, io.BytesIO(b"a" * (MAX_IMAGE_SIZE + 1)), "large.jpg")
    assert response.status_code == 413
    assert response.json == {"error": "File too large. Maximum size is 16 MB."}


def test_accepts_valid_image_upload(client):
    response = post_image(client, make_image_bytes(), "image.png")
    assert response.status_code == 200
    assert response.data == b"[]"
