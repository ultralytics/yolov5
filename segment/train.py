# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Forward the legacy segmentation training command to the YOLOv5 package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolov5.segment.train import main, parse_opt  # noqa: E402

if __name__ == "__main__":
    main(parse_opt())
