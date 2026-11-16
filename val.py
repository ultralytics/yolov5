# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Forward the legacy validation command to the YOLOv5 package."""

from yolov5.val import main, parse_opt

if __name__ == "__main__":
    main(parse_opt())
