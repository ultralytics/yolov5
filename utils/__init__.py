# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
utils/ initialization
"""


def notebook_init():
    # For use in YOLOv5 notebooks
    from IPython.display import clear_output  # to display images

    from utils.torch_utils import select_device  # YOLOv5 imports

    clear_output()
    device, string = select_device()
    print(f'Setup done! Using {string.rstrip()}')
