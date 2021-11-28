# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization
"""


def notebook_init():
    # For  notebooks
    print('Checking setup...')

    import os
    import shutil
    from IPython import display  # to display images and clear console output
    from utils.general import emojis, check_requirements
    from utils.torch_utils import select_device  # imports

    check_requirements(('psutil',))
    import psutil

    # System
    gb = 1 / 1000 ** 3  # bytes to GiB
    gib = 1 / 1024 ** 3  # bytes to GB
    ram = psutil.virtual_memory().total
    total, used, free = shutil.disk_usage("/")
    display.clear_output()
    s = f'{os.cpu_count()} CPUs, {ram * gib:.1f} GB RAM, {(total - free) * gb:.1f}/{total * gb:.1f} GB disk'

    select_device(newline=False)
    print(emojis(f'Setup complete ✅ ({s})'))
    return display
