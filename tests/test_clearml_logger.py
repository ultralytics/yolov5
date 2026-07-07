# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import utils.loggers as loggers

clearml_utils = importlib.import_module("utils.loggers.clearml.clearml_utils")


class Opt(SimpleNamespace):
    def __contains__(self, key):
        """Support logger membership checks used by the training options object."""
        return hasattr(self, key)


def make_opt(tmp_path):
    return Opt(
        data="data/coco128.yaml",
        evolve=False,
        exist_ok=False,
        name="exp",
        noplots=True,
        project=Path("runs/train"),
        resume=False,
        save_dir=tmp_path,
        sync_bn=False,
    )


@pytest.mark.parametrize("legacy_error", [False, True])
def test_clearml_not_configured_warns_and_disables_logger(monkeypatch, tmp_path, legacy_error):
    """ClearML auth/config errors should not crash training startup."""
    MissingConfigError = ValueError if legacy_error else type("MissingConfigError", (ValueError,), {})

    class Task:
        @staticmethod
        def init(**kwargs):
            raise MissingConfigError("ClearML configuration could not be found")

    monkeypatch.setattr(loggers, "clearml", object())
    monkeypatch.setattr(clearml_utils, "clearml", object())
    monkeypatch.setattr(clearml_utils, "MissingConfigError", MissingConfigError, raising=False)
    monkeypatch.setattr(clearml_utils, "Task", Task, raising=False)
    warnings = []
    monkeypatch.setattr(loggers.LOGGER, "warning", warnings.append)

    logger = loggers.Loggers(
        save_dir=tmp_path, opt=make_opt(tmp_path), hyp={}, logger=logging.getLogger(), include=("clearml",)
    )

    assert logger.clearml is None
    assert any("ClearML is installed but not configured, skipping ClearML logging" in w for w in warnings)


def test_clearml_unexpected_value_error_surfaces(monkeypatch, tmp_path):
    """Unexpected legacy ClearML ValueErrors should still fail startup."""

    class Task:
        @staticmethod
        def init(**kwargs):
            raise ValueError("Task type 'bad' not supported")

    monkeypatch.setattr(loggers, "clearml", object())
    monkeypatch.setattr(clearml_utils, "clearml", object())
    monkeypatch.setattr(clearml_utils, "MissingConfigError", ValueError, raising=False)
    monkeypatch.setattr(clearml_utils, "Task", Task, raising=False)

    with pytest.raises(ValueError, match="Task type"):
        loggers.Loggers(
            save_dir=tmp_path, opt=make_opt(tmp_path), hyp={}, logger=logging.getLogger(), include=("clearml",)
        )
