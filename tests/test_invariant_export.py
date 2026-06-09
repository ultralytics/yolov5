import pytest
import subprocess
import os
import tempfile


@pytest.mark.parametrize("malicious_path", [
    "/tmp/model; touch /tmp/pwned",
    "/tmp/model$(touch /tmp/pwned)",
    "/tmp/model`touch /tmp/pwned`",
    "valid_model.pt",
])
def test_shell_injection_via_model_path(malicious_path):
    """Invariant: Shell metacharacters in model paths must never execute arbitrary commands."""
    # Clean up any evidence file before test
    pwned_marker = "/tmp/pwned"
    if os.path.exists(pwned_marker):
        os.remove(pwned_marker)

    # Attempt to run export.py with a malicious model path
    # We expect it to fail gracefully (file not found, invalid model, etc.)
    # but NEVER execute injected commands
    result = subprocess.run(
        ["python", "export.py", "--weights", malicious_path, "--include", "onnx"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # The critical security invariant: no injected command was executed
    assert not os.path.exists(pwned_marker), (
        f"Shell injection succeeded with payload: {malicious_path!r}. "
        f"Arbitrary command execution detected!"
    )