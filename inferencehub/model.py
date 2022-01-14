from utils.general import non_max_suppression
import torch
import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(self, network_pkl):
        super().__init__()
        # network parameters
        self.network_pkl = network_pkl

        # model loading
        self.model = attempt_load(network_pkl, map_location="cpu")
        self.add_module("model", self.model)

    def forward(self, x: dict, input_parameters: dict) -> torch.tensor:
       pred = self.model(x, augment=input_parameters.get('augment', False), visualize=input_parameters.get('visualize', False))[0]
       pred = non_max_suppression(pred,
                                input_parameters.get('conf_thres', 0.25),
                                input_parameters.get('iou_thres', 0.45),
                                input_parameters.get('classes'),
                                input_parameters.get('agnostic_nms', False),
                                max_det=input_parameters.get('max_det', 1000))
       return pred

def get_model(weights_path=None, map_location="cpu", model_initialization_parameters={}) -> torch.nn.Module:
    return ModelWrapper(network_pkl=weights_path)
