# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize(
    "malicious_path",
    [
        "/tmp/model; touch /tmp/pwned",
        "/tmp/model$(touch /tmp/pwned)",
        "/tmp/model`touch /tmp/pwned`",
        "valid_model.pt",
    ],
)
def test_shell_injection_via_model_path(malicious_path):
    """Invariant: Shell metacharacters in model paths must never execute arbitrary commands."""
    pwned_marker = "/tmp/pwned"
    if os.path.exists(pwned_marker):
        os.remove(pwned_marker)

    subprocess.run(
        [sys.executable, "export.py", "--weights", malicious_path, "--include", "onnx"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert not os.path.exists(pwned_marker), (
        f"Shell injection succeeded with payload: {malicious_path!r}. Arbitrary command execution detected!"
    )


def test_export_edgetpu_no_shell_true():
    """Invariant: export_edgetpu() must never call subprocess.run with shell=True."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Stub out heavy ML dependencies so export.py can be imported
    heavy_mods = [
        "torch",
        "torch.nn",
        "torch.utils",
        "torch.utils.mobile_optimizer",
        "pandas",
        "ultralytics",
        "ultralytics.utils",
        "ultralytics.utils.patches",
        "models",
        "models.experimental",
        "models.yolo",
        "utils",
        "utils.dataloaders",
        "utils.general",
        "utils.torch_utils",
        "utils.segment",
        "utils.segment.general",
    ]
    stubs = {}
    for mod in heavy_mods:
        stubs[mod] = MagicMock()
    stubs["torch.utils.mobile_optimizer"].optimize_for_mobile = MagicMock()

    with patch.dict("sys.modules", stubs):
        # Remove cached module if already imported
        sys.modules.pop("export", None)
        import export as export_module

    shell_true_calls = []

    def mock_run(*args, **kwargs):
        if kwargs.get("shell") is True:
            shell_true_calls.append((args, kwargs))
        result = MagicMock()
        result.returncode = 0
        result.stdout = b"2.1.0"
        return result

    with patch.object(export_module, "subprocess") as mock_subprocess, patch.object(
        export_module, "platform"
    ) as mock_platform:
        mock_platform.system.return_value = "Linux"
        mock_subprocess.DEVNULL = subprocess.DEVNULL
        mock_subprocess.PIPE = subprocess.PIPE

        def track_run(*args, **kwargs):
            if kwargs.get("shell") is True:
                shell_true_calls.append((args, kwargs))
            result = MagicMock()
            result.returncode = 0
            result.stdout = b"2.1.0"
            return result

        mock_subprocess.run.side_effect = track_run
        try:
            export_module.export_edgetpu(Path("/tmp/fake_model.pt"))
        except Exception:
            pass  # Only care that shell=True was never used

    assert not shell_true_calls, f"export_edgetpu() called subprocess.run with shell=True: {shell_true_calls}"
