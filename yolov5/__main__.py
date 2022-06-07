import fire

from yolov5.detect import run as detect
from yolov5.export import run as export
from yolov5.train import run as train
from yolov5.utils.benchmarks import run as benchmark
from yolov5.val import run as val


def main():
    fire.Fire({
        'train': train,
        'detect': detect,
        'val': val,
        'export': export,
        'benchmark': benchmark
    })


if __name__ == "__main__":
    main()
