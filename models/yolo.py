# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """
        Initialize the YOLOv5 detection layer with specified class counts, anchors, input channels, and inplace
        operations.

        Args:
            nc (int): Number of object classes that the model should detect.
            anchors (tuple): Tuple consisting of anchor box dimensions.
            ch (tuple): Tuple containing the number of input channels for each detection layer.
            inplace (bool): If True, performs operations in-place for memory efficiency.

        Returns:
            None

        Example:
            ```python
            # Initialize a Detect layer with 80 classes, given anchors, and input channels for each detection layer
            detect_layer = Detect(nc=80, anchors=anchors, ch=(256, 512, 1024))
            ```
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """
        Processes input data through YOLOv5 detection layers and adjusts tensor shapes for detection outputs.

        Args:
            x (list[torch.Tensor]): Input data for detection, where each tensor corresponds to a detection layer output
                and has shape (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH).

        Returns:
            (list[torch.Tensor]): Processed detection outputs that contain bounding box coordinates, objectness scores,
                and class scores. Each tensor has shape (BATCH_SIZE, NUM_ANCHORS * HEIGHT * WIDTH, OUTPUT_DIMENSIONS).

        Note:
            The function differentiates between training and inference modes. During inference, it dynamically adjusts the
            grid and anchor grid shapes if needed. The method also accounts for whether the model is a `Segment` instance
            (for both boxes and masks) or a `Detect` instance (for boxes only).

        Example:
            ```python
            model = Detect(nc=80, anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], ch=[128, 256, 512])
            x = [torch.rand(1, 128, 20, 20), torch.rand(1, 256, 10, 10), torch.rand(1, 512, 5, 5)]
            outputs = model(x)
            ```
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """
        Generate a grid of coordinates and scaled anchor boxes for YOLOv5 detection layers.

        Args:
            nx (int): Number of grid cells along the x-axis.
            ny (int): Number of grid cells along the y-axis.
            i (int): Index of the detection layer.
            torch_1_10 (bool): Compatibility check for torch versions below 1.10.0. Default is `check_version(torch.__version__, "1.10.0")`.

        Returns:
            (torch.Tensor, torch.Tensor): Tuple containing:
                - grid (torch.Tensor): Grid coordinates tensor with shape (1, num_anchors, ny, nx, 2).
                - anchor_grid (torch.Tensor): Scaled anchor boxes tensor with shape (1, num_anchors, ny, nx, 2).

        Example:
            ```python
            grid, anchor_grid = self._make_grid(20, 20, 0)
            ```

        Note:
            The function creates a coordinate grid and scales anchor boxes according to the strides and anchors specified for the YOLOv5 model.
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """
        Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments.

        Args:
            nc (int): Number of classes for the segmentation model, default is 80 classes.
            anchors (tuple): Tuple representing anchor box dimensions.
            nm (int): Number of masks, default is 32.
            npr (int): Number of protos, default is 256.
            ch (tuple): Tuple representing the number of input channels for each image.
            inplace (bool): Whether to use inplace operations to save memory, default is True.

        Returns:
            None

        Example:
            ```python
            segment_head = Segment(nc=80, anchors=anchors, nm=32, npr=256, ch=(256, 512, 1024), inplace=True)
            ```

        Note:
            Required initialization for YOLOv5 segmentation models, integrating detection and mask prediction capabilities.
        """
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """
        Process input through the YOLOv5 segment head, returning detection boxes and mask prototypes.

        Args:
            x (list[torch.Tensor]): Input image tensor of shape (N, C, H, W).

        Returns:
            tuple: Tuple containing:
                - torch.Tensor: Concatenated detection results with shape (N, M, `5 + nc + nm`), where nc represents the
                    number of classes and nm the number of masks. During training, the shape is (N, M, 6).
                - torch.Tensor: Mask prototypes with shape (N, 32, H, W).

        Example:
            ```python
            model = Segment(nc=80, anchors=[...], nm=32, npr=256, ch=[...])
            inputs = [...]  # Provide a list of input tensors
            detections, prototypes = model(inputs)
            ```
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """
        Performs a forward pass through the YOLOv5 base model, optionally profiling and visualizing the process.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is the batch size, C is the number of channels,
                H is the height, and W is the width.
            profile (bool): If True, profiles the forward pass. Default is False.
            visualize (bool): If True, visualizes feature maps. Default is False.

        Returns:
            (torch.Tensor | tuple[torch.Tensor, ...]): Model output. If in inference mode, returns a single torch.Tensor.
                If training, returns a tuple of tensors.

        Example:
            ```python
            model = BaseModel()
            input_tensor = torch.randn(1, 3, 640, 640)
            output = model(input_tensor)
            ```

        Note:
            Profiling outputs can be used to understand the compute requirements of the model layers, while visualization
            aids in examining the intermediate feature maps.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the YOLOv5 model, enabling optional profiling and feature visualization.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is the batch size, C is the number of
                channels, H is the height, and W is the width.
            profile (bool): If True, profiles the forward pass, logging time and memory consumption. Default is False.
            visualize (bool | str): If True or a path string, visualizes feature maps and saves the output. Default is False.

        Returns:
            (torch.Tensor | tuple[torch.Tensor, ...]): Model output tensor during inference, or a tuple of tensors during
                training. During inference, returns a single tensor of shape (N, M, 85) or (N, M, 6) depending on the model.
                During training, returns a tuple containing tensors from intermediate layers.

        Example:
            ```python
            # Initialize model
            model = BaseModel()

            # Generate random input tensor
            input_tensor = torch.randn(1, 3, 640, 640)

            # Perform forward pass with profiling and visualization
            output = model._forward_once(input_tensor, profile=True, visualize='path/to/save')

            # Perform standard forward pass
            output = model._forward_once(input_tensor)
            ```

        Note:
            Enable profiling to understand the computational resources required by each layer. Visualization is
            beneficial to analyze intermediate feature maps.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        Profiles a single model layer's performance by measuring GFLOPs, execution time, and parameter count.

        Args:
            m (nn.Module): The neural network layer to be profiled.
            x (torch.Tensor): Input tensor to the layer being profiled, with shape (N, C, H, W) where N is the batch size,
                C is the number of channels, H is the height, and W is the width.
            dt (list): A list to which the execution times (in milliseconds) for the layer will be appended.

        Returns:
            None

        Note:
            The function uses the `thop` library for FLOPs computation and `time_sync()` for measuring the time. Make sure
            `thop` is installed for computing GFLOPs, otherwise, GFLOPs will be zero. The input tensor `x` might be copied
            for the final layer to avoid in-place operation issues.

        Example:
            ```python
            model = BaseModel()
            dt = []
            model._profile_one_layer(layer, input_tensor, dt)
            ```
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """
        Fuses `Conv2d` and `BatchNorm2d` layers in the model to improve inference speed and efficiency.

        This method modifies the model in-place by fusing the convolutional and batch normalization layers, which can
        result in faster inference times and reduced memory usage due to consolidating separate layers into a single
        operation. Generally applied after model training but before deployment for inference.

        Args:
            None

        Returns:
            None: The model is modified in-place without returning any value.

        Example:
            ```python
            model = BaseModel()
            model.fuse()
            ```

        Note:
            After calling `fuse()`, the model will no longer have separate batch normalization layers. This could make
            further fine-tuning or training less effective unless the changes are undone.
        """
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """
        Prints summary information about the YOLOv5 base model, including layer details and computational cost.

        Args:
            verbose (bool): If True, prints a detailed layer-by-layer summary of the model. Default is False.
            img_size (int): The size of the input image to the model, used for FLOPs computation. Default is 640.

        Returns:
            None

        Example:
            ```python
            model = BaseModel()
            model.info(verbose=True, img_size=640)
            ```

        Note:
            The detailed summary includes information about each layer's type, number of parameters, computational cost (FLOPs),
            and execution time. Use this information to understand the model's structure and performance implications better.
        """
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """
        Apply a function to transform the BaseModel's tensors.

        Args:
            fn (Callable): Function that applies a transformation such as to(), cpu(), cuda(), half() to the model's
                tensors.

        Returns:
            (BaseModel): The model with the transformation function applied to its tensors.

        Note:
            This method excludes parameters or registered buffers from being transformed. It targets tensors
            specifically related to the model's detect or segmentation layers, such as stride, grid, or anchor_grid.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """
        Initialize the YOLOv5 detection model with configuration, input channels, number of classes, and anchors.

        Args:
            cfg (str | dict): Configuration file or dictionary. For example, 'yolov5s.yaml'.
            ch (int): Number of input channels. Default is 3.
            nc (int | None): Number of classes. If provided, it overrides the number specified in the configuration.
            anchors (list | None): List of anchors. If provided, it overrides the anchors specified in the configuration.

        Returns:
            (None): Initializes the model with the specified parameters and configurations.

        Example:
            ```python
            model = DetectionModel(cfg='yolov5s.yaml', ch=3, nc=80, anchors=None)
            ```

        Note:
            The model structure is defined according to the configuration file (or dictionary). The strides and anchors
            are built and initialized based on the input configuration. If the number of classes or anchors are provided,
            they override the respective values in the configuration file. The model's weights and biases are also initialized
            during this process.
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):

            def _forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Perform a forward pass through the YOLOv5 detection model during training or inference.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is the batch size, C is the number of channels, H is
                the height, and W is the width.
            augment (bool): If True, applies augmented inference.
            profile (bool): If True, profiles the forward pass. Default is False.
            visualize (bool): If True, visualizes feature maps. Default is False.

        Returns:
            (tuple[torch.Tensor, torch.Tensor] | torch.Tensor): If not training, returns concatenated detections with shape
                (N, M, 85/6) and feature maps list. If in training mode, returns a tuple of both raw outputs and feature outputs.

        Example:
            ```python
            model = DetectionModel(cfg='yolov5s.yaml')
            input_tensor = torch.randn(1, 3, 640, 640)
            output = model(input_tensor)
            ```

        Note:
            When exporting the model, concatenated detections are returned alone. This is useful for further processing and
            optimization.
        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """
        Combine augmented inference results obtained from varying scales and flips.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is the batch size, C is the number of channels,
                H is the height, and W is the width.

        Returns:
            (torch.Tensor): Aggregated detection results after applying scale and flip augmentations, matching the input
                image size.

        Notes:
            This method performs augmented inference by scaling and flipping the input image tensor 'x' through predefined
            scales and flips. Each augmented version is processed through the model, and the results are combined to generate
            a comprehensive detection output.

        Example:
            ```python
            model = DetectionModel(cfg='yolov5s.yaml')
            input_tensor = torch.randn(1, 3, 640, 640)
            augmented_output = model._forward_augment(input_tensor)
            ```
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """
        De-scale augmented predictions to original image size, adjusting coordinates for any flips applied.

        Args:
            p (torch.Tensor): Augmented predictions tensor with shape (N, M, 6) where N is the batch size and M is the number
                of predictions.
            flips (int | None): Indicates the type of flip applied to the image augmentation (2 for up-down, 3 for
                left-right, None for no flip).
            scale (float): Scaling factor applied during image augmentation.
            img_size (tuple[int, int]): Original image size as a tuple (height, width).

        Returns:
            (torch.Tensor): De-scaled predictions tensor with the same shape as input, adjusted to the original image size.

        Note:
            This function is used to reverse any augmentations applied during inference, such as scaling and flipping,
            to ensure the predictions are correctly mapped to the original image dimensions. If `self.inplace` is True,
            it operates in-place on the predictions tensor; otherwise, it creates new tensors for adjusted coordinates.

        Example:
            ```python
            model = DetectionModel(cfg="yolov5s.yaml")
            augmented_pred = model.forward_augmented(input_tensor)
            descaled_pred = model._descale_pred(augmented_pred, flips=3, scale=0.8, img_size=(640, 640))
            ```
        """
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """
        Clip augmented inference tails for YOLOv5 models.

        Args:
            y (list[torch.Tensor]): List of augmented inference results, each a torch.Tensor with shape (B, N, 85), where
                B is the batch size, N is the number of detections, and 85 includes (x, y, w, h, conf, cls).

        Returns:
            (list[torch.Tensor]): Clipped augmented inference results with adjusted shapes to remove tails.

        Note:
            This method specifically adjusts the first and last layers based on the number of detection layers and grid points,
            improving inference accuracy by removing extraneous detections from augmented inputs.

        Example:
            ```python
            y_augmented = [
                torch.randn(1, 1200, 85),  # Example augmented result 1
                torch.randn(1, 300, 85)  # Example augmented result 2
            ]
            y_clipped = detection_model._clip_augmented(y_augmented)
            ```
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """
        Initialize biases for the YOLOv5 'Detect' layer.

        Args:
            cf (torch.Tensor | None): Optional tensor of class frequencies with shape (C,) where C is the number of classes.
                If not provided, biases are initialized using a default strategy.

        Returns:
            (torch.Tensor): Updated biases for the detection model's convolutional layers.

        Note:
            For more information, refer to the paper here: https://arxiv.org/abs/1708.02002 (Section 3.3).

        Example:
            ```python
            # Assuming 'model' is an instance of the YOLOv5 DetectionModel
            class_frequencies = torch.tensor([10, 20, 30, 40])  # example class frequencies
            model._initialize_biases(class_frequencies)
            ```

            If `cf` is not provided, the function uses default values to initialize biases:
            ```python
            model._initialize_biases()
            ```
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """
        Initializes a YOLOv5 segmentation model with a configuration file, input channels, classes, and anchors.

        Args:
            cfg (str): Path to the model configuration file (e.g., 'yolov5s-seg.yaml').
            ch (int): Number of input channels.
            nc (int | None): Number of classes for segmentation. Overrides the number of classes defined in the
                configuration file if provided.
            anchors (list | None): List of anchor box scales. If provided, overrides the anchors specified in the configuration file.

        Returns:
            (None): This constructor does not return a value. It initializes the segmentation model.

        Example:
            ```python
            model = SegmentationModel(cfg='yolov5s-seg.yaml', ch=3, nc=21, anchors=[[10,13], [16,30], [33,23]])
            ```

        Note:
            This model inherits from the `DetectionModel` class and adapts the YOLO architecture to include mask prediction
            for segmentation tasks.
        """
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """
        Initialize a YOLOv5 Classification model with a given configuration file or a pre-defined model.

        Args:
            cfg (str | None): Path to the model configuration file, specifying the model's architecture and parameters.
                If None, a pre-defined model must be provided.
            model (nn.Module | None): A pre-defined model which can be used instead of loading from a configuration file.
                If None, the model will be built based on the configuration file specified in 'cfg'.
            nc (int): Number of classes for the classification task. Determines the size of the output layer.
            cutoff (int): Layer cutoff index for the model. Default is 10. Determines where the backbone ends and the classifier starts.

        Notes:
            This method sets up the YOLOv5 classification model using either a configuration file or a pre-defined model. The
            specified number of classes ('nc') determines the output layer size and is crucial for correct classification.

        Examples:
            ```python
            # Initialize with a configuration file
            model = ClassificationModel(cfg='yolov5s.yaml', nc=1000)

            # Initialize with a pre-defined model
            pre_defined_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model = ClassificationModel(model=pre_defined_model, nc=1000)
            ```
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """
        Create a classification model from a detection model by truncating it at a specified layer and adding a
        classification head.

        Args:
            model (nn.Module): YOLOv5 detection model to be converted.
            nc (int): Number of classes for the classification model. Default is 1000.
            cutoff (int): Layer index at which to split the detection model for creating the classification model. Default is 10.

        Returns:
            None

        Note:
            This function modifies the input model in place by truncating it at the specified `cutoff` layer and adding
            a classification head.

        Example:
            ```python
            detection_model = DetectionModel()
            classification_model = ClassificationModel()
            classification_model._from_detection_model(detection_model, nc=5, cutoff=8)
            ```
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a YAML configuration file."""
        self.model = None


def parse_model(d, ch):
    """
    Parses the YOLO model configuration and constructs model layers accordingly.

    Args:
        d (dict): The YOLO model configuration dictionary containing backbone and head definitions, among other parameters.
        ch (list[int]): List of input channels, typically specified by the model's input configuration.

    Returns:
        (tuple[List[nn.Module], List[int]]):
            - List of created model layers.
            - List of indices specifying which layers produce output that needs to be saved during the forward pass.

    Example:
        ```python
        config = {
            'anchors': [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
            'nc': 80,
            'depth_multiple': 0.33,
            'width_multiple': 0.50,
            'backbone': [
                [-1, 1, 'Conv', [64, 6, 2, 2]],  # from, number, module, args
                [-1, 1, 'BottleneckCSP', [64, 3]],
            ],
            'head': [
                [-1, 1, 'Detect', [80, 1, 1]],
            ],
        }
        ch = [3]  # RGB input
        layers, save = parse_model(config, ch)
        ```

    Note:
        This function supports various model components including `Conv`, `BottleneckCSP`, `Detect`, and others. The function will log
        the details of each layer as it is created, including the from layer, number of repetitions, number of parameters, module type,
        and arguments.
    """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
