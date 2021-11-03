# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
utils/ initialization
"""


def init_notebook():
    # For use in YOLOv5 notebooks
    from IPython.display import Image, clear_output  # to display images
    from utils.torch_utils import select_device  # YOLOv5 imports

    clear_output()
    return select_device(), Image
