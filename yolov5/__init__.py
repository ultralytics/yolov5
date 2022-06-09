import yolov5.detect as detect
import yolov5.export as export
import yolov5.train as train
import yolov5.utils.benchmarks as benchmark
import yolov5.val as val

# Version format VERSION = f"{_MAJOR}.{_MINOR}.{_PATCH}{_SUFFIX}" https://semver.org/#is-v123-a-semantic-version
__version__ = "0.8.0"
__all__ = ["train", "detect", "val", "export", "benchmark"]
