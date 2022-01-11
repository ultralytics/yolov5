 
from models.experimental import attempt_load
import torch


test_file = "./data/images/bus.jpg"

def get_model(weights_path=None, map_location="cpu") -> torch.nn.Module:
    return attempt_load(weights_path, map_location=map_location)
