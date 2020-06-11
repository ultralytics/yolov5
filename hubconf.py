"""File for accessing YOLOv5 via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, channels=3, classes=80)
"""

dependencies = ['torch', 'yaml']
import torch

from models.yolo import Model
from utils import google_utils


def create(name, pretrained, channels, classes):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    model = Model('models/%s.yaml' % name, channels, classes)
    if pretrained:
        ckpt = '%s.pt' % name  # checkpoint filename
        google_utils.attempt_download(ckpt)  # download if not found locally
        state_dict = torch.load(ckpt)['model'].state_dict()
        state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].numel() == v.numel()}  # filter
        model.load_state_dict(state_dict, strict=False)  # load
    return model


def yolov5s(pretrained=False, channels=3, classes=80):
    """YOLOv5-small model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5s', pretrained, channels, classes)


def yolov5m(pretrained=False, channels=3, classes=80):
    """YOLOv5-medium model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5m', pretrained, channels, classes)


def yolov5l(pretrained=False, channels=3, classes=80):
    """YOLOv5-large model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5l', pretrained, channels, classes)


def yolov5x(pretrained=False, channels=3, classes=80):
    """YOLOv5-xlarge model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov5x', pretrained, channels, classes)
