# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
from utils.tal.anchor_generator import generate_anchors, dist2bbox

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class V6Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.reg_max = 16 + 1
        self.no = nc + self.reg_max * 4 + 1  # number of outputs per anchor
        self.grid = torch.empty(0)  # init
        self.anchor_grid = torch.empty(0)  # init
        self.stride_grid = torch.empty(0)  # init
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.shape = (0, 0)  # initial grid shape

        c2, c3 = 32, max(ch[0], self.no - 4)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc + 1, 1)) for x in ch)
        self.proj_conv = nn.Conv2d(self.reg_max, 1, 1, bias=False).requires_grad_(False)
        self.proj_conv.weight.data[:] = nn.Parameter(torch.arange(0, self.reg_max).float().view(1, self.reg_max, 1, 1))
        self.initialize_biases()

    def initialize_biases(self):
        for seq in self.cv2:
            m = seq[-1]
            m.bias.data[:] = 1.0
            m.weight.data[:] = 0.0
        for seq in self.cv3:
            m = seq[-1]
            m.bias.data[:] = -math.log((1 - 1e-2) / 1e-2)
            m.weight.data[:] = 0.0

    def print_biases(self):
        for i, seq in enumerate(self.cv2):
            m = seq[-1]
            print(f'cv2-{i} weight/bias {m.weight.data.mean()}, {m.bias.data.mean()}')
        for i, seq in enumerate(self.cv3):
            m = seq[-1]
            print(f'cv3-{i} weight/bias {m.weight.data.mean()}, {m.bias.data.mean()}')


    def forward(self, x):
        self.print_biases()
        b = x[0].shape[0]
        for i in range(self.nl):
            # y = self.cv1[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        y = torch.cat([xi.view(b, self.no, -1) for xi in x], dim=-1)
        y = y.permute(0, 2, 1).contiguous()  # (b, grids, 85)
        bbox, conf, cls = y.split((self.reg_max * 4, 1, self.nc), -1)
        if self.training:
            return x, conf, cls, bbox
        anchors, strides = generate_anchors(x, torch.tensor([8, 16, 32]), 5.0, 0.5, device=x[0].device, is_eval=True)

        # DFL box
        dbox = bbox.reshape([b, -1, 4, self.reg_max]).permute(0, 3, 2, 1)  # b, reg_max+1, 4, grids
        dbox = self.proj_conv(F.softmax(dbox, dim=1)).view(b, 4, -1)  # b, 4, grids
        dbox = dbox.permute(0, 2, 1).contiguous()  # b, grids, 4

        dbox = dist2bbox(dbox, anchors, box_format="xywh")  # (b, grids, 4)
        dbox *= strides
        return torch.cat([dbox, conf.sigmoid(), cls.sigmoid()], -1).permute(0, 2, 1).contiguous(), (x, conf, cls, bbox)


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.grid = torch.empty(0)  # init
        self.anchor_grid = torch.empty(0)  # init
        self.stride_grid = torch.empty(0)  # init
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.shape = (0, 0)  # initial grid shape
        c2, c3 = 64, max(ch[0], self.no - 4)  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(
            # Conv(x, c2, 3), Conv(c2, c2, 3), Conv(c2, 4, 1, act=False)) for x in ch)
            # Conv(x, c2, 3), Conv(c2, c2, 3), Conv(c2, 64, 1), DFL(64 // 4)) for x in ch)
            Conv(x, c2, 3), Conv(c2, c2, 3), DFL(64 // 4)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(
            Conv(x, c3, 3), Conv(c3, c3, 3), Conv(c3, self.no - 4, 1, act=False)) for x in ch)

    def forward(self, x):
        for i in range(self.nl):
            # y = self.cv1[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x

        bs, _, ny, nx = x[0].shape  # x(bs,85,20,20)
        y = torch.cat([x.view(bs, self.no, x.shape[2] * x.shape[3]) for x in x], 2)  # cat all outputs
        if self.dynamic or self.shape != (ny, nx):  # build grids
            self._make_grids(nx, ny)

        if isinstance(self, Segment):  # (boxes + masks)
            xy, wh, conf, mask = y.split((2, 2, self.nc + 1, self.no - self.nc - 5), 1)
            xy = xy.sigmoid() * (1.6 * self.stride_grid) + self.grid  # xy
            wh = (0.2 + wh.sigmoid() * 4.8) * self.anchor_grid
            y = torch.cat((xy, wh, conf.sigmoid(), mask), 1)
        else:  # Detect (boxes only)
            xy, wh, conf = y.sigmoid().split((2, 2, self.nc + 1), 1)
            xy = xy * (1.6 * self.stride_grid) + self.grid  # xy
            wh = (0.2 + wh * 4.8) * self.anchor_grid
            y = torch.cat((xy, wh, conf), 1)

        return (y,) if self.export else (y, x)

    def _make_grids(self, nx=20, ny=20, torch_1_10=check_version(torch.__version__, '1.10.0')):
        grids, d, t = [], self.anchors[0].device, self.anchors[0].dtype  # grids, device, type
        for i, stride in enumerate(self.stride):
            nyi, nxi = (int(x * self.stride[0] / stride) for x in (ny, nx))
            shape = 1, 2, nyi, nxi  # grid shape
            y, x = torch.arange(nyi, device=d, dtype=t), torch.arange(nxi, device=d, dtype=t)
            yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)
            grid_xy = torch.stack((xv, yv), 0).expand(shape) - 0.3  # i.e. y = 1.6 * x - 0.3
            grid_wh = self.anchors[i].view((1, 2, 1, 1)).expand(shape)
            grid_stride = torch.ones(shape, device=d, dtype=t)
            grids.append(torch.cat((grid_xy, grid_wh, grid_stride), 1).view(1, 6, nyi * nxi) * stride)
        self.grid, self.anchor_grid, self.stride_grid = torch.cat(grids, 2).chunk(3, 1)
        self.shape = (ny, nx)


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
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
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            for k in 'stride', 'anchor_grid', 'stride_grid', 'grid':
                x = getattr(m, k)
                setattr(m, k, list(map(fn, x))) if isinstance(x, (list, tuple)) else setattr(m, k, fn(x))
        elif isinstance(m, V6Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, V6Detect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, V6Detect)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            if not isinstance(m, V6Detect):
                self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
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
        # de-scale predictions following augmented inference (inverse operation)
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
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3 for 4-81 splits
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        m = self.model[-1]  # Detect() module
        ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            # a[-1].bn.bias.data[2:4] = -1.38629  # wh = 0.25 + (x - 1.38629).sigmoid() * 3.75
            a[-1].bn.bias.data[2:4] = -1.60944  # wh = 0.2 + (x - 1.60944).sigmoid() * 4.8
            b[-1].bn.bias.data[0] = math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[-1].bn.bias.data[1:m.nc + 1] = ncf  # cls


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus,
            BottleneckCSP, C1, C2, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {V6Detect, Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov8m-2-upsample-tal.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)
    model.eval()

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
