# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression tests for the Flask REST API authentication guard (security invariant V-002).

Run with:
    pytest utils/flask_rest_api/test_restapi.py -v
"""

import os
import signal
import subprocess
import time

import pytest
import requests

# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flask_server():
    """Start the production Flask server with an API key and yield its process."""
    restapi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "restapi.py")
    env = {**os.environ, "API_KEY": "test-secret-key"}

    proc = subprocess.Popen(
        ["python", restapi_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )

    # Wait for the server to be ready
    time.sleep(3)

    yield proc

    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

ENDPOINT = "http://127.0.0.1:5000/v1/object-detection/yolov5s"
FAKE_IMAGE = ("test.jpg", b"fake_image_data", "image/jpeg")


@pytest.mark.parametrize(
    "auth_header",
    [
        # No authentication at all (primary exploit scenario)
        None,
        # Malformed / invalid token
        "Bearer invalid_token_123",
        # Expired-style JWT (no valid signature)
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE1MTYyMzkwMjJ9.old_signature",
        # Empty bearer value
        "Bearer ",
        # Plausible-looking but wrong key
        "Bearer valid_but_unauthorized",
    ],
)
def test_protected_endpoint_rejects_unauthenticated_requests(flask_server, auth_header):
    """Security invariant V-002: protected endpoints must reject unauthenticated requests with 401/403."""
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header

    try:
        response = requests.post(ENDPOINT, files={"image": FAKE_IMAGE}, headers=headers, timeout=5)
        assert response.status_code in {401, 403}, (
            f"Endpoint accepted an unauthenticated request (auth_header={auth_header!r}). "
            f"Got {response.status_code} instead of 401/403."
        )
    except requests.exceptions.ConnectionError:
        pytest.fail("Could not connect to the Flask server. Ensure the fixture started correctly.")


def test_authenticated_request_is_not_rejected_with_401_403(flask_server):
    """A request carrying the correct API key must not be turned away with 401/403."""
    headers = {"X-API-Key": "test-secret-key"}

    try:
        response = requests.post(ENDPOINT, files={"image": FAKE_IMAGE}, headers=headers, timeout=5)
        # The server may return 400 (bad image), 404 (model not loaded), etc.
        # What it must NOT return is 401 or 403.
        assert response.status_code not in {401, 403}, (
            f"A correctly authenticated request was rejected with {response.status_code}."
        )
    except requests.exceptions.ConnectionError:
        pytest.fail("Could not connect to the Flask server. Ensure the fixture started correctly.")
