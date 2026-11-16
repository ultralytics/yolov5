# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Forward PyTorch Hub entry points to the YOLOv5 package."""

if __name__ == "__main__":
    from runpy import run_module

    run_module("yolov5.hubconf", run_name="__main__")
else:
    from yolov5.hubconf import *  # noqa: F403
