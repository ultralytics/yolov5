# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization
"""


def notebook_init():
    # For  notebooks
    print('Checking setup...')

    import os
    import psutil
    import shutil
    from IPython import display  # to display images and clear console output
    from utils.general import emojis
    from utils.torch_utils import select_device  # imports

    # System
    gb = 1 / 1024 ** 3  # bits to GiB
    ram = psutil.virtual_memory().total
    total, used, free = shutil.disk_usage("/")
    display.clear_output()
    s = f'{os.cpu_count()} CPUs, {ram * gb:.1f} GB RAM, {used * gb:.1f}/{total * gb:.1f} GB disk'

    select_device(newline=False)
    print(emojis(f'Setup complete ✅ ({s})'))
    return display
