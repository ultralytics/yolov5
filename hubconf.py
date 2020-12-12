"""File for accessing YOLOv5 via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, channels=3, classes=80)
"""

from pathlib import Path

import torch

from models.yolo import Model
from utils.general import set_logging
from utils.google_utils import attempt_download

dependencies = ['torch', 'yaml']
set_logging()


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
    config = Path(__file__).parent / 'models' / f'{name}.yaml'  # model.yaml path
    try:
        model = Model(config, channels, classes)
        if pretrained:
            fname = f'{name}.pt'  # checkpoint filename
            attempt_download(fname)  # download if not found locally
            ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
            model.load_state_dict(state_dict, strict=False)  # load
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names  # set class names attribute
            # model = model.autoshape()  # for PIL/cv2/np inputs and NMS
        return model

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache maybe be out of date, try force_reload=True. See %s for help.' % help_url
        raise Exception(s) from e


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


def custom(model='path/to/model.pt'):
    """YOLOv5-custom model from https://github.com/ultralytics/yolov5

    Arguments (3 format options):
        model (str): 'path/to/model.pt'
        model (dict): torch.load('path/to/model.pt')
        model (nn.Module): 'torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    if isinstance(model, str):
        model = torch.load(model)  # load checkpoint
    if isinstance(model, dict):
        model = model['model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    return hub_model


if __name__ == '__main__':
    model = create(name='yolov5s', pretrained=True, channels=3, classes=80)  # pretrained example
    # model = custom(model='path/to/model.pt')  # custom example
    model = model.autoshape()  # for PIL/cv2/np inputs and NMS

    # Verify inference
    from PIL import Image

    imgs = [Image.open(x) for x in Path('data/images').glob('*.jpg')]
    results = model(imgs)
    results.show()
    results.print()
