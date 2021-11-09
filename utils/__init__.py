# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization
"""


def notebook_init():
    # For YOLOv5 notebooks
    print('Checking setup...')
    from IPython import display  # to display images and clear console output

    from utils.general import emojis
    from utils.torch_utils import select_device  # YOLOv5 imports

    display.clear_output()
    select_device(newline=False)
    print(emojis('Setup complete ✅'))
    return display
