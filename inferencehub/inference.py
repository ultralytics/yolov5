import torch

def preprocess_function(image_np: np.ndarray) -> torch.tensor:
    # transform to pytorch tensor
    img_torch = torch.from_numpy(image_np)
    return img_torch


def postprocess_function(image_torch: torch.tensor) -> torch.tensor:
    return image_torch
