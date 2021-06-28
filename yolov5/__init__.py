import importlib.util
from pathlib import Path

import sys

PIP_PACKAGE_NAME = 'yolov5'
PIP_PACKAGE_ROOT: Path = Path(__file__).parent
GIT_REPOSITORY_ROOT: Path = Path(__file__).parent.parent


def yolo_in_python_env() -> bool:
    spec = importlib.util.find_spec(PIP_PACKAGE_NAME)
    return spec is not None


# appending pip package root directory to path to support legacy way of running script
def mount_yolo_if_required() -> None:
    if yolo_in_python_env():
        return
    sys.path.append(GIT_REPOSITORY_ROOT.as_posix())
