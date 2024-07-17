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
        Initializes the YOLOv5 Detect layer with class count, anchors, channels, and inplace operations.

        Args:
            nc (int, optional): Number of classes. Default is 80.
            anchors (tuple, optional): Anchor box dimensions, typically specified for each detection layer. Default is ().
            ch (tuple, optional): Number of input channels for each detection layer. Default is ().
            inplace (bool, optional): If True, operations are done inplace. Default is True.

        Returns:
            None

        Example:
            ```python
            detect_layer = Detect(nc=80, anchors=(), ch=(256, 512, 1024), inplace=True)
            ```

        Note:
            This function initializes detection heads in the YOLOv5 model, setting up convolution layers, grids, and
            anchor grids required for object detection inference.
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
        Processes input through detection layers, reshaping and applying convolution for YOLOv5 inference.

        Args:
            x (list[torch.Tensor]): List of feature maps from backbone with shape (B, C, H, W) where B is the batch
                size, C is the number of channels, and H and W are height and width.

        Returns:
            (list[torch.Tensor]): List of processed detections, each a torch Tensor with shape (B, N, D) where B
                is the batch size, N is the number of detections, and D is the dimensions of each detection
                (e.g., bounding box coordinates, objectness score, class probabilities).

        Note:
            This method applies a series of convolutions to transform the input feature maps into detection
            outputs. It also handles reshaping and permutation to align with YOLOv5's output format. During
            inference, additional steps are performed to compute final object locations and dimensions.
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
        Generate a mesh grid for anchor boxes with torch version compatibility for detection models.

        Args:
            nx (int): Number of grid cells along the x-axis.
            ny (int): Number of grid cells along the y-axis.
            i (int): Index of the detection layer for which the grid is being generated.
            torch_1_10 (bool): Indicator whether the torch version is at least 1.10.0 for meshgrid compatibility.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): A tuple containing two tensors:
                - grid (torch.Tensor): The generated grid with shape (1, num_anchors, ny, nx, 2), containing xy coordinates.
                - anchor_grid (torch.Tensor): The anchor grid scaled by the stride, with shape (1, num_anchors, ny, nx, 2).

        Example:
            ```python
            detector = Detect()
            grid, anchor_grid = detector._make_grid(20, 20, 0)
            ```

        Note:
            The function ensures compatibility with different torch versions by using appropriate meshgrid indexing options.
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
        Initializes YOLOv5 Segment head with parameters for masks, prototypes, class count, anchors, and channels.

        Args:
            nc (int): Number of classes for the segmentation model (default is 80).
            anchors (tuple): Tuple of anchor box dimensions for the segmentation model.
            nm (int): Number of masks for the segmentation (default is 32).
            npr (int): Number of prototypes for the masks (default is 256).
            ch (tuple): Tuple of input channels for each detection layer.
            inplace (bool): If True, use in-place operations for layer computations (default is True).

        Returns:
            None

        Example:
            ```python
            segment_head = Segment(nc=80, anchors=anchors, nm=32, npr=256, ch=[512, 256, 128], inplace=True)
            ```
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
        Processes input through the network, returning detections and prototypes.

        Args:
            x (list[torch.Tensor]): List of input tensors corresponding to different detection layers, each with shape
                (B, C, H, W), where B is batch size, C is number of channels, H and W are height and width.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - `detection` (torch.Tensor): The detection output tensor with shape (B, N, 85), where B is batch size, N is
                  the number of detections.
                - `prototypes` (torch.Tensor): The prototype masks tensor produced by the network with shape (B, P, H', W'),
                  where B is batch size, P is the number of prototypes, and H' and W' correspond to height and width.

         Example:
            ```python
            import torch
            from ultralytics import YOLOv5

            # Initialize model
            model = YOLOv5.Segment()

            # Generate dummy input
            x = [torch.randn(1, 3, 640, 640) for _ in range(3)]

            # Forward pass
            detection, prototypes = model.forward(x)
            ```

        Note:
            During inference (evaluation mode), detection outputs are post-processed to generate final bounding boxes and classes.
            In training mode, the outputs are not processed.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the YOLOv5 model, optionally profiling and visualizing features.

        Args:
            x (torch.Tensor): Input data tensor with shape (N, C, H, W).
            profile (bool): Whether to profile execution time of each layer. Defaults to False.
            visualize (bool): Whether to store and visualize feature maps. Defaults to False.

        Returns:
            (torch.Tensor | tuple): In training mode, returns predictions as tuples with shapes (N, 3, H, W, no).
            In inference mode, returns a single tensor with shape (N, M, no), where M is the number of predicted
            objects after non-maximum suppression (NMS).

        Example:
            ```python
            model = BaseModel()
            input_tensor = torch.randn(1, 3, 640, 640)
            output = model.forward(input_tensor, profile=True, visualize=True)
            ```

        Note:
            - In training mode, the method returns unprocessed predictions for each scale, suitable for loss calculation.
            - In inference mode, non-maximum suppression is applied to refine predictions.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
        Execute a forward pass through the YOLOv5 model layers with optional profiling and visualization.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is the batch size, C is the number
                of channels, and H and W are the height and width of the input image, respectively.
            profile (bool): If True, profiles the execution time for each layer. Defaults to False.
            visualize (bool): If True, stores and visualizes feature maps. Defaults to False.

        Returns:
            (torch.Tensor): Model output tensor with shape depending on whether the model is in training or
            inference mode.
                - In training mode: Returns a list of tensors for each detection layer, each tensor has shape
                  (N, 3, H, W, no), where `no` is the number of outputs per anchor.
                - In inference mode: If not exporting, returns a tuple with a single tensor of shape (N, M, no),
                  where M is the number of predicted objects.
                - If exporting: Returns a tensor of shape (N, M, no).

        Example:
            ```python
            model = BaseModel()
            input_tensor = torch.randn(1, 3, 640, 640)  # Generate a random input tensor
            output = model._forward_once(input_tensor, profile=True, visualize=True)
            ```

        Note:
            This method conducts a single-scale inference or training pass through the model. Depending on the mode
            (training or inference), the method behaves differently. In training mode, it returns unprocessed
            predictions for each detection layer. In inference mode, non-maximum suppression (NMS) is typically
            applied after this method to refine predictions.
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
        Profiles a single model layer's GFLOPs, parameters, and execution time within the YOLOv5 model.

        Args:
            m (nn.Module): The model layer to be profiled.
            x (torch.Tensor): Input tensor passed to the model layer, with shape (N, C, H, W).
            dt (list[float]): List to record execution times of the profiled layer.

        Returns:
            None: The function updates the `dt` list with the execution time of the layer in milliseconds.

        Example:
            ```python
            model = BaseModel()
            layer = nn.Conv2d(3, 16, 3, 1)  # Example layer
            input_tensor = torch.randn(1, 3, 640, 640)  # Example input
            execution_times = []

            model._profile_one_layer(layer, input_tensor, execution_times)
            ```

        Note:
            - Profiling is done for the purpose of understanding the computational load (GFLOPs) and time taken per layer within
              the YOLOv5 model.
            - If the `thop` library is not available, FLOPs computation will not be performed.
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
        Fuses Conv2d and BatchNorm2d layers in the model to optimize inference speed.

        This method modifies the model in place by merging Conv2d and BatchNorm2d layers into single Conv2d
        layers where applicable. This can significantly improve inference speed and reduce memory usage.

        Returns:
            None

        Example:
            ```python
            model = BaseModel()
            model.fuse()
            ```

        Note:
            After fusing layers, the forward method of fused layers is updated to `forward_fuse`, optimizing
            the execution path.
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
        Display model summary, including layer details and computational complexity for a specified image size.

        Args:
            verbose (bool): If True, prints a detailed summary including information about each layer. Defaults to False.
            img_size (int | tuple[int]): Size of the input image as an integer (for square images) or tuple (H, W).
                Defaults to 640.

        Returns:
            (None): This function does not return any value. It directly prints the model summary to the console.

        Example:
            ```python
            model = BaseModel()
            model.info(verbose=True, img_size=640)
            ```

        Note:
            Ensure that the `verbose` parameter is set to True for a comprehensive layer-by-layer summary. The image size should
            be supplied based on the expected input size for the model.
        """
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """
        Apply a function to the model and its layer parameters, including specific modifications for Detect and Segment
        layers.

        Args:
            fn (function): A function to apply to the model's tensors.

        Returns:
            (torch.nn.Module): The module with applied transformations.

        Note:
            The function is particularly useful for operations like converting tensors to a target device
            (e.g., CUDA, CPU) or altering their precision (e.g., float16). The Detect layer's stride and grid
            parameters, as well as the Segment layer's anchor grids, are specifically modified to ensure consistency
            after such transformations.
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
        Initializes YOLOv5 model using the specified config, input channels, class count, and custom anchors.

        Args:
            cfg (str | dict): Model configuration, either a path to a YAML config file or a configuration dictionary.
            ch (int): Number of input channels. Defaults to 3.
            nc (int | None): Number of classes. If provided, overrides the value in the YAML file/config dictionary. Defaults to None.
            anchors (list[int] | None): Custom anchors. If provided, overrides the anchors defined in the YAML file/config
                dictionary. Defaults to None.

        Returns:
            None

        Example:
            ```python
            from ultralytics.models.yolo import DetectionModel

            # Initialize model with path to YAML config
            model1 = DetectionModel(cfg="yolov5s.yaml")

            # Initialize model with configuration dictionary
            cfg_dict = {"nc": 80, "depth_multiple": 0.33, "width_multiple": 0.50}
            model2 = DetectionModel(cfg=cfg_dict, ch=3, nc=80)
            ```

        Note:
            If `cfg` is a dictionary, it should include the necessary parameters such as `nc`, `depth_multiple`, and `width_multiple`.
            During initialization, the model configuration from the YAML file or dictionary is parsed, and the internal model
            structure is built accordingly. This includes defining the detection layers and adjusting anchors and strides.
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
        Perform forward pass through the YOLOv5 detection model for training or inference, with options for
        augmentation, profiling, and visualization.

        Args:
            x (torch.Tensor): Input tensor with a shape of (N, C, H, W), where N is the batch size, C is the number of channels,
                H is the height, and W is the width.
            augment (bool): If True, performs augmented inference. Defaults to False.
            profile (bool): If True, profiles the execution time of each layer. Defaults to False.
            visualize (bool): If True, stores and visualizes feature maps. Defaults to False.

        Returns:
            (torch.Tensor | tuple): Depending on the mode, returns either:
                - In training mode: tuple containing predictions for each scale with shapes (N, 3, H, W, no).
                - In inference mode: tensor with shape (N, M, no), where M is the number of predicted objects after
                  non-maximum suppression.
                - When exporting: tuple containing concatenated inference output tensor and intermediate feature maps.

        Example:
            ```python
            model = DetectionModel(cfg="yolov5s.yaml", ch=3, nc=80)
            input_tensor = torch.randn(1, 3, 640, 640)
            output = model.forward(input_tensor, augment=False, profile=True, visualize=False)
            ```

        Note:
            This method adapts to training and inference modes, with different return types based on the operational mode.
            During training mode, it returns raw predictions across various scales for loss calculation, whereas in inference
            mode, non-maximum suppression (NMS) is applied to refine predictions.
        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """
        Performs augmented inference by processing input across different scales and flips, merging the outputs.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where N is batch size, C is number of channels,
                H and W are height and width.

        Returns:
            (torch.Tensor): Merged output tensor after multi-scale and flip augmentations, with shape (N, M, no),
                where N is batch size, M is the number of predictions, and no is the number of output features.

        Example:
            ```python
            model = DetectionModel(cfg='yolov5s.yaml')
            input_tensor = torch.randn(1, 3, 640, 640)
            output = model._forward_augment(input_tensor)
            ```

        Note:
            The function processes the input using different scales (1, 0.83, 0.67) and flips (None, horizontal),
            descaling predictions before merging. This helps to improve model robustness and accuracy
            during inference.
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
        Adjusts predictions for augmented inference by de-scaling and correcting for flips or image size changes.

        Args:
            p (torch.Tensor): Predictions tensor with shape (..., N) where N indicates prediction attributes like
                bounding box coordinates, confidence score, etc.
            flips (int | None): Specifies flip mode. `2` for vertical flip, `3` for horizontal flip, and `None` for no flip.
            scale (float): Scale factor used during augmentation.
            img_size (tuple[int, int]): Original image dimensions as (height, width).

        Returns:
            (torch.Tensor): Adjusted predictions tensor with the same shape as input, de-scaled and de-flipped appropriately.

        Note:
            If inplace operations are enabled, the adjustments are applied directly on the tensor. Otherwise, new tensors are
            created for the adjusted values to avoid modifying the original input.
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
        Clip augmented inference tails for YOLOv5 models, adjusting predictions from the first and last layers.

        Args:
            y (list[torch.Tensor]): List of tensors, where each tensor represents detections from augmented inference across different layers.

        Returns:
            (list[torch.Tensor]): Modified list of tensors with clipped augmented inference tails.

        Notes:
            This function helps to discard the augmented tails by adjusting predictions from the first and last layers,
            which might otherwise introduce artifacts due to the augmentation process.
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
        Initialize biases for the YOLOv5 Detect module using specified or default bias adjustments.

        Args:
            cf (torch.Tensor | None): Optional tensor representing class frequencies for bias initialization. The shape should be
                (N,), where N is the number of classes. If not provided, default adjustments are applied based on the number of
                classes and image dimensions.

        Returns:
            (torch.Tensor): Updated biases for the model with shape (N, M), where N is the number of anchors and M is the number of
                outputs per anchor.

        Note:
            The function calculates the biases based on principles from https://arxiv.org/abs/1708.02002, section 3.3. If class
            frequencies (`cf`) are not provided, default bias adjustments are made. Adjustments primarily ensure that objectness and
            class biases are reasonably initialized for effective training.

        Example:
            ```python
            from ultralytics.yolov5 import DetectionModel
            import torch

            # Initialize model
            model = DetectionModel(cfg="yolov5s.yaml")

            # Optional class frequencies tensor
            class_frequencies = torch.tensor([100, 150, 200])

            # Initialize biases
            model._initialize_biases(cf=class_frequencies)
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
        Initializes a YOLOv5 segmentation model with configurable parameters.

        Args:
            cfg (str): Path to the configuration file containing model architecture and parameters. Defaults to "yolov5s-seg.yaml".
            ch (int): Number of input channels. Defaults to 3.
            nc (int | None): Number of classes. If provided, overrides the number of classes specified in the cfg file.
            anchors (list | None): List of anchor points. If provided, overrides the anchor configuration in the cfg file.

        Returns:
            (None): Initializes various components of the SegmentationModel class.

        Example:
            ```python
            from ultralytics import SegmentationModel
            model = SegmentationModel()
            ```

        Note:
            The initialization includes setting up model layers, anchors, and other configurations based on the provided
            or default configuration file.
        """
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """
        Initializes a YOLOv5 classification model with either a configuration file or a pre-built model, specifying the
        number of classes and a cutoff layer index.

        Args:
            cfg (str | None): Path to the model configuration file, or None if using `model`.
            model (torch.nn.Module | None): Pre-built torch model, or None if using `cfg`.
            nc (int): Number of output classes, default is 1000.
            cutoff (int): Index of the cutoff layer, default is 10.

        Returns:
            None

        Example:
            ```python
            # Initializing from a configuration file
            model = ClassificationModel(cfg='yolov5-class-config.yaml', nc=1000, cutoff=10)

            # Initializing from an existing model
            model = ClassificationModel(model=prebuilt_model, nc=1000, cutoff=10)
            ```

        Note:
            This model can be extended or customized by modifying the configuration file or the pre-built model.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """
        Perform a transformation from a YOLOv5 detection model to a classification model.

        Args:
            model (DetectionModel): A pre-trained YOLOv5 detection model.
            nc (int): Number of classes for the classification model. Default is 1000.
            cutoff (int): Index to slice the model's layers up to the classification layer. Default is 10.

        Returns:
            None. The function modifies the model in place.

        Notes:
            This function takes a detection model and transforms it into a classification model by slicing the model layers
            at the specified cutoff point and adding a classification layer with the specified number of classes.
            - If the input model is wrapped by `DetectMultiBackend`, it unwraps the model to get the underlying YOLOv5 model.
            - Constructs a `Classify` layer, replacing the final detection layer with this new classification layer.

        Example:
            ```python
            from ultralytics import YOLOv5

            # Load a pre-trained detection model
            detection_model = YOLOv5.load('yolov5s.pt')

            # Create a classification model from detection model
            classification_model = YOLOv5.ClassificationModel()
            classification_model._from_detection_model(detection_model, nc=1000, cutoff=10)
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
        """
        Perform initialization and parsing from a YOLOv5 configuration file.

        Args:
            cfg (str): Path to the YOLOv5 YAML configuration file.

        Returns:
            None. The function modifies the model in place utilizing the defined configuration parameters.

        Notes:
            This function reads a YOLOv5 YAML configuration file and constructs the classification model accordingly. It sets the
            appropriate channels, layers, and output classes based on the parsed configuration data.
        """
        self.model = None


def parse_model(d, ch):
    """
    Parses YOLOv5 model architecture from a configuration dictionary and initializes its layers.

    Args:
        d (dict): Dictionary containing model configuration. Must include keys: "anchors", "nc", "depth_multiple",
            "width_multiple", and optionally "activation" and "channel_multiple".
        ch (list[int]): List of input channels for each layer.

    Returns:
        (tuple[nn.Sequential, list[int]]): A tuple containing:
            - `model` (nn.Sequential): The constructed YOLOv5 model based on the configuration.
            - `save` (list[int]): List of layers whose outputs should be preserved during the forward pass.

    Example:
        ```python
        from pathlib import Path
        import yaml

        # Load model configuration YAML
        with open(Path('yolov5s.yaml'), 'r') as file:
            model_config = yaml.safe_load(file)

        # Parse model and initialize
        model, save = parse_model(model_config, ch=[3])
        ```
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
