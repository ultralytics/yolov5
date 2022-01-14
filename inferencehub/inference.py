import torch
import numpy as np
from utils.datasets import LoadStreams, LoadImages

def preprocess_function(input_pre: dict,  model, input_parameters: dict) -> dict:
    img = torch.from_numpy(np.array(input_pre))
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img

# You need to pass a torch.tensor back.
def postprocess_function(image_torch: torch.tensor) -> torch.tensor:
    return image_torch
