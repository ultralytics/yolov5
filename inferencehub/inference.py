import torch
import numpy as np
from utils.augmentations import letterbox
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size
import cv2


def preprocess_function(input_pre: dict, model, input_parameters: dict) -> dict:
    # Validate image size
    imgsz = input_parameters.get('imgsz', 640)
    stride = int(model.model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    input_parameters['imgsz'] = imgsz
    nparr = np.asarray(input_pre, dtype="uint8")

    # Padded resize
    img = letterbox(nparr, imgsz, stride=stride)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img

# You need to pass a torch.tensor back.
def postprocess_function(image_torch: torch.tensor) -> torch.tensor:
    return image_torch
