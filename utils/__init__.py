from pathlib import Path

import torch
from PIL import ImageFont

FILE = Path(__file__).absolute()
ROOT = FILE.parents[1]  # yolov5/ dir

# Check YOLOv5 Annotator font
font = 'Arial.ttf'
try:
    ImageFont.truetype(font)
except Exception as e:  # download if missing
    url = "https://ultralytics.com/assets/" + font
    print(f'Downloading {url} to {ROOT / font}...')
    torch.hub.download_url_to_file(url, str(ROOT / font))
