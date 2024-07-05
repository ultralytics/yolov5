# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # official model
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # from branch
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # local repo
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """
    Creates or loads a YOLOv5 model, with options for pretrained weights and model customization.

    Args:
        name (str): Model name (e.g., 'yolov5s') or path to the model checkpoint (e.g., 'path/to/best.pt').
        pretrained (bool): If True, loads pretrained weights into the model. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper to the model for extended input support.
            Default is True.
        verbose (bool): If True, prints detailed information to the console. Default is True.
        device (str | torch.device | None): Device to use for the model parameters, specified as a string, torch.device,
            or None to use the default device. Default is None.

    Returns:
        YOLOv5 model (torch.nn.Module): Loaded or created YOLOv5 model instance.

    Raises:
        Exception: If there is an error in loading or creating the model, an exception is raised with a helpful message.
            See https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading for troubleshooting.

    Example:
    ```python
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # from a specific branch
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # local repository
    ```

    Note:
    - This function is typically used through `torch.hub.load` interface.
    - Pretrained models are specifically configured for 3-channel images and 80-class detection tasks.
    - For further assistance, visit the PyTorch Hub models documentation at https://pytorch.org/hub/ultralytics_yolov5.
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(ROOT / "requirements.txt", exclude=("opencv-python", "tensorboard", "thop"))
    name = Path(name)
    path = name.with_suffix(".pt") if name.suffix == "" and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning(
                            "WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. "
                            "You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224)."
                        )
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning(
                            "WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. "
                            "You will not be able to run inference with this model."
                        )
                    else:
                        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            cfg = list((Path(__file__).parent / "models").rglob(f"{path.stem}.yaml"))[0]  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt["model"].names) == classes:
                    model.names = ckpt["model"].names  # set class names attribute
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # reset to default
        return model.to(device)

    except Exception as e:
        help_url = "https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading"
        s = f"{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help."
        raise Exception(s) from e


def custom(path="path/to/model.pt", autoshape=True, _verbose=True, device=None):
    """
    Loads a custom or local YOLOv5 model from a given path with optional autoshaping and device specification.

    Args:
        path (str): Path to the custom model file (e.g., 'path/to/model.pt').
        autoshape (bool): If True, applies YOLOv5 .autoshape() wrapper to the model (default is True).
        _verbose (bool): If True, prints all information to screen (default is True).
        device (str | torch.device | None): Device to use for model parameters, e.g., 'cuda' or 'cpu' (default is None).

    Returns:
        torch.nn.Module: Loaded YOLOv5 model.

    Raises:
        Exception: Raises an exception with a helpful message if model loading fails.

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'custom_model.pt')
        ```

        ```python
        # Loading a model with autoshape disabled
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'custom_model.pt', autoshape=False)
        ```

    Notes:
        For more information and detailed usage, see the PyTorch Hub documentation at:
        https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading
    """
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiates the YOLOv5-nano model with options for pretraining, input channels, class count, autoshaping,
    verbosity, and device.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels. Defaults to 3.
        classes (int): Number of model classes. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper to the model. Defaults to True.
        _verbose (bool): If True, prints all information to the screen. Defaults to True.
        device (str | torch.device | None): Device to use for model parameters (e.g., 'cpu', 'cuda'). Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5-nano model, optionally with pretrained weights and/or autoshaping.

    Notes:
        For more information on model loading and usage, refer to the official [PyTorch Hub models](https://pytorch.org/hub/ultralytics_yolov5).

    Examples:
        ```python
        import torch

        # Load a pretrained YOLOv5-nano model with default settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

        # Load a YOLOv5-nano model with custom settings
        model_custom = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=False, channels=1, classes=10,
                                      autoshape=False, _verbose=False, device='cpu')
        ```
    """
    return _create("yolov5n", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates YOLOv5-small model with options for pretraining, input channels, class count, autoshaping, verbosity, and
    device.

    Args:
        pretrained (bool, optional): Flag to load pretrained weights into the model. Defaults to True.
        channels (int, optional): Number of input channels. Defaults to 3.
        classes (int, optional): Number of model classes. Defaults to 80.
        autoshape (bool, optional): Whether to apply YOLOv5 .autoshape() wrapper to the model for pre-processing. Defaults to True.
        _verbose (bool, optional): Flag to print detailed information to the screen. Defaults to True.
        device (str | torch.device | None, optional): Device to use for model parameters, specified as a string (e.g., "cpu",
           "cuda:0") or torch.device object. If None, default device will be used. Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5-small model ready for inference or training.

    Note:
        For further information on using PyTorch Hub models, refer to:
        https://pytorch.org/hub/ultralytics_yolov5

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        ```
    """
    return _create("yolov5s", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiates the YOLOv5-medium model with customizable pretraining, channel count, class count, autoshaping,
    verbosity, and device.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): Apply YOLOv5 .autoshape() wrapper to the model for handling various input formats. Default is True.
        _verbose (bool, optional): Whether to print all information to the screen. Default is True.
        device (str | torch.device | None, optional): Device specification to use for model parameters (e.g., 'cpu', 'cuda'). Default is None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5-medium model.

    Usage Example:
    ```python
    import torch

    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Load YOLOv5-medium model with default parameters
    ```
    """
    return _create("yolov5m", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates YOLOv5-large model with options for pretraining, channels, classes, autoshaping, verbosity, and device
    selection.

    Args:
        pretrained (bool): Load pretrained weights into the model. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): Apply YOLOv5 .autoshape() wrapper to model. Default is True.
        _verbose (bool): Print all information to screen. Default is True.
        device (str | torch.device | None): Device to use for model parameters, e.g., 'cpu', 'cuda', or a torch.device instance. Default is None.

    Returns:
        YOLOv5 model (torch.nn.Module).

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        ```

    Notes:
        For additional details, refer to the PyTorch Hub models documentation:
        https://pytorch.org/hub/ultralytics_yolov5
    """
    return _create("yolov5l", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiates the YOLOv5-xlarge model with customizable pretraining, channel count, class count, autoshaping,
    verbosity, and device.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels for the model. Defaults to 3.
        classes (int): Number of classes for the model. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper to the model for varied input types. Defaults to True.
        _verbose (bool): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None): Device on which to load the model (e.g., 'cpu', 'cuda'). Defaults to None.

    Returns:
        (torch.nn.Module): The instantiated YOLOv5-xlarge model.

    Note:
        For additional guidance and examples, refer to the PyTorch Hub documentation:
        https://pytorch.org/hub/ultralytics_yolov5

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Load YOLOv5-xlarge model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=False)  # Load model without pretrained weights
        ```
    """
    return _create("yolov5x", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates YOLOv5-nano-P6 model with options for pretraining, channels, classes, autoshaping, verbosity, and device.

    Args:
        pretrained (bool, optional): If True, loads pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): If True, applies the YOLOv5 .autoshape() wrapper to the model. Default is True.
        _verbose (bool, optional): If True, prints all information to screen. Default is True.
        device (str | torch.device | None, optional): Device to use for model parameters. Can be a string
            representing the device ('cpu', 'cuda'), a torch.device object, or None to select automatically. Default is None.

    Returns:
        torch.nn.Module: The created YOLOv5-nano-P6 model.

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n6')  # load pretrained YOLOv5-nano-P6 model
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5n6')  # load model from master branch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n6.pt')  # load custom/local model
        ```

    Note:
        For more detailed model loading options, visit the PyTorch Hub models page:
        https://pytorch.org/hub/ultralytics_yolov5
    """
    return _create("yolov5n6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiate the YOLOv5-small-P6 model with options for pretraining, input channels, number of classes, autoshaping,
    verbosity, and device selection.

    Args:
        pretrained (bool): If True, loads pretrained weights. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of object detection classes. Default is 80.
        autoshape (bool): If True, applies YOLOv5 .autoshape() wrapper to the model, allowing for varied input formats.
            Default is True.
        _verbose (bool): If True, prints detailed information during model loading. Default is True.
        device (str | torch.device | None): Device specification for model parameters (e.g., 'cpu', 'cuda', or torch.device).
            Default is None, which selects an available device automatically.

    Returns:
        torch.nn.Module: The YOLOv5-small-P6 model instance.

    Usage:
        ```python
        import torch

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s6')  # load from a specific branch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/yolov5s6.pt')  # custom/local model
        model = torch.hub.load('.', 'custom', 'path/to/yolov5s6.pt', source='local')  # local repo
        ```
    """
    return _create("yolov5s6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates YOLOv5-medium-P6 model with options for pretraining, channel count, class count, autoshaping, verbosity, and
    device.

    Args:
        pretrained (bool): If True, loads pretrained weights. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): Apply YOLOv5 .autoshape() wrapper to the model for file/URI/PIL/cv2/np inputs and NMS. Default is True.
        _verbose (bool): If True, prints detailed information to the screen. Default is True.
        device (str | torch.device | None): Device to use for model parameters. Default is None, which uses the best available device.

    Returns:
        torch.nn.Module: The YOLOv5-medium-P6 model.

    Refer to the PyTorch Hub models documentation: https://pytorch.org/hub/ultralytics_yolov5 for additional details.

    Example:
    ```python
    import torch

    # Load YOLOv5-medium-P6 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
    ```

    Notes:
    - The model can be loaded with pre-trained weights for better performance on specific tasks.
    - The autoshape feature simplifies input handling by allowing various popular data formats.
    """
    return _create("yolov5m6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Instantiates the YOLOv5-large-P6 model with customizable pretraining, channel and class counts, autoshaping,
    verbosity, and device selection.

    Args:
        pretrained (bool, optional): If True, load pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): If True, apply YOLOv5 .autoshape() wrapper to the model for input flexibility.
            Default is True.
        _verbose (bool, optional): If True, print all information to the screen. Default is True.
        device (str | torch.device | None, optional): Device to use for model parameters, e.g., 'cpu', 'cuda', or
            torch.device. If None, automatically selects the best available device. Default is None.

    Returns:
        models.yolo.DetectionModel: A YOLOv5-large-P6 model configured with the specified options.

    Usage:
    ```python
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')  # load pretrained YOLOv5-large-P6 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=False, channels=1, classes=10)  # customized
    ```
    """
    return _create("yolov5l6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates the YOLOv5-xlarge-P6 model with options for pretraining, number of input channels, class count, autoshaping,
    verbosity, and device selection.

    Args:
        pretrained (bool): Whether to load pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels for the model. Defaults to 3.
        classes (int): Number of classes for the model. Defaults to 80.
        autoshape (bool): Whether to apply YOLOv5 .autoshape() wrapper to the model. Defaults to True.
        _verbose (bool): Whether to print all information to the screen. Defaults to True.
        device (str | torch.device | None): The device to be used for model parameters. Defaults to None.

    Returns:
        torch.nn.Module: An instance of the YOLOv5-xlarge-P6 model, optionally pretrained.

    Note:
        For more details, refer to the PyTorch Hub models documentation: https://pytorch.org/hub/ultralytics_yolov5

    Examples:
        ```python
        import torch
        model = torch.hub.load("ultralytics/yolov5", "yolov5x6")
        ```
    """
    return _create("yolov5x6", pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5s", help="model name")
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    imgs = [
        "data/images/zidane.jpg",  # filename
        Path("data/images/zidane.jpg"),  # Path
        "https://ultralytics.com/images/zidane.jpg",  # URI
        cv2.imread("data/images/bus.jpg")[:, :, ::-1],  # OpenCV
        Image.open("data/images/bus.jpg"),  # PIL
        np.zeros((320, 640, 3)),
    ]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()
