# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compatibility exports and CLI forwarding for legacy YOLOv5 model paths."""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from runpy import run_module

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    run_module("yolov5.models.yolo", run_name="__main__")
else:
    from yolov5.models.yolo import *  # noqa: F403
