from yolov5.detect import run as detect
from yolov5.export import run as export
from yolov5.train import run as train
from yolov5.utils.benchmarks import run as benchmark
from yolov5.val import run as val

__all__ = ["train", "detect", "val", "export", "benchmark"]
