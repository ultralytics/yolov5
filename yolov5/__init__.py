# Add the path to the lowest yolov5 folder of the hierarchy (the one containing this __init__) to sys.path
# to avoid the error "ModuleNotFoundError: No module named 'models'" in torch.load()

import os
import sys

yolov5_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, yolov5_dir_path)