import fire

from yolov5 import benchmark, detect, export, train, val


def _PrintResult(component_trace, verbose=False):
    pass


# Patch: fire cli displays help text if the object is not printable
fire.core._PrintResult = _PrintResult


def main():
    fire.Fire({
        'train': train.run,
        'detect': detect.run,
        'val': val.run,
        'export': export.run,
        'benchmark': benchmark.run})


if __name__ == "__main__":
    main()
