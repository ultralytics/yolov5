import fire

from yolov5 import benchmark, detect, export, train, val


def main():
    fire.Fire({
        'train': train.run,
        'detect': detect.run,
        'val': val.run,
        'export': export.run,
        'benchmark': benchmark.run})


if __name__ == "__main__":
    main()
