import sys
import logging
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import argparse
import torch.nn.functional as F
from copy import deepcopy
from torch.autograd import Variable
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import torch.nn.init as init
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        d = [] # mask training
        self.training |= self.export

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # if not self.training:  # inference
            if self.stride is not None:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()

                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5  + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # d.append(y.clone().view(bs, -1, self.no))
                z.append(y.view(bs, -1, self.no))

        return x if self.stride is None else (torch.cat(z, 1), x, z)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def log2(x):
  """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
  ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
  if x.is_cuda:
    ln2 = ln2.cuda()
  return torch.log(x) / ln2

class SamePad2d(nn.Module):
  """Mimics tensorflow's 'SAME' padding.
  """

  def __init__(self, kernel_size, stride):
    super(SamePad2d, self).__init__()
    self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
    self.stride = torch.nn.modules.utils._pair(stride)

  def forward(self, input):
    in_width = input.size()[2]
    in_height = input.size()[3]
    out_width = math.ceil(float(in_width) / float(self.stride[0]))
    out_height = math.ceil(float(in_height) / float(self.stride[1]))
    pad_along_width = ((out_width - 1) * self.stride[0] +
                       self.kernel_size[0] - in_width)
    pad_along_height = ((out_height - 1) * self.stride[1] +
                        self.kernel_size[1] - in_height)
    pad_left = math.floor(pad_along_width / 2)
    pad_top = math.floor(pad_along_height / 2)
    pad_right = pad_along_width - pad_left
    pad_bottom = pad_along_height - pad_top
    return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

  def __repr__(self):
    return self.__class__.__name__

def pyramid_roi_align(inputs, pool_size, image_shape):
  boxes = inputs[0]
  feature_maps = inputs[1:]

  # Assign each ROI to a level in the pyramid based on the ROI area.
  y1, x1, y2, x2 = boxes.chunk(4, dim=1)
  h = y2 - y1
  w = x2 - x1

  # Equation 1 in the Feature Pyramid Networks paper. Account for
  # the fact that our coordinates are normalized here.
  # e.g. a 224x224 ROI (in pixels) maps to P4
  image_area = Variable(torch.FloatTensor([float(image_shape[0] * image_shape[1])]), requires_grad=False)

  if boxes.is_cuda:
    image_area = image_area.cuda()

  roi_level = 0 * log2(torch.sqrt(h * w) / (128 / torch.sqrt(image_area)))
  roi_level = roi_level.round().int()
  # roi_level = roi_level.clamp(0, 0)
  #roi_level = 0

  # Loop through levels and apply ROI pooling to each. P2 to P5.
  pooled = []
  box_to_level = []

  for i, level in enumerate(range(0, 1)):
    ix = roi_level == level
    if not ix.any():
      continue
    ix = torch.nonzero(ix)[:, 0]
    level_boxes = boxes[ix.data, :]

    # Keep track of which box is mapped to which level
    box_to_level.append(ix.data)

    # Stop gradient propogation to ROI proposals
    level_boxes = level_boxes.detach()

    ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False).int()

    if level_boxes.is_cuda:
      ind = ind.cuda()

    feature_maps[i] = feature_maps[i].unsqueeze(0)  # CropAndResizeFunction needs batch dimension
    #(height, width)
    # pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
    pooled_features = CropAndResizeFunction.apply(feature_maps[i], level_boxes, ind, pool_size, pool_size)
    pooled.append(pooled_features)

  # Pack pooled features into one tensor
  pooled = torch.cat(pooled, dim=0)

  # Pack box_to_level mapping into one array and add another
  # column representing the order of pooled boxes
  box_to_level = torch.cat(box_to_level, dim=0)

  # Rearrange pooled features to match the order of the original boxes
  _, box_to_level = torch.sort(box_to_level)
  pooled = pooled[box_to_level, :, :]

  return pooled


class Mask_Head(nn.Module):
  def __init__(self, depth, num_classes):
    super(Mask_Head, self).__init__()
    self.depth = depth
    self.num_classes = num_classes
    self.padding = SamePad2d(kernel_size=3, stride=1)
    self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
    self.bn1 = nn.BatchNorm2d(256, eps=0.001)
    self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
    self.bn2 = nn.BatchNorm2d(256, eps=0.001)
    self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
    self.bn3 = nn.BatchNorm2d(256, eps=0.001)
    self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
    self.bn4 = nn.BatchNorm2d(256, eps=0.001)
    self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
    self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x, rois, image_shape):
    pool_size = 14
    x = pyramid_roi_align([rois] + x, pool_size, image_shape)
    x = self.conv1(self.padding(x))
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(self.padding(x))
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(self.padding(x))
    x = self.bn3(x)
    x = self.relu(x)
    x = self.conv4(self.padding(x))
    x = self.bn4(x)
    x = self.relu(x)
    x = self.deconv(x)
    x = self.relu(x)
    x = self.conv5(x)
    x = self.sigmoid(x)
    return x

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, training=True):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if training:
            if nc and nc != self.yaml['nc']:
                logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
                self.yaml['nc'] = nc  # override yaml value
            if anchors:
                logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
                self.yaml['anchors'] = round(anchors)  # override yaml value
        else:
            self.yaml['nc'] = nc
            if anchors:
                self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        #Mask head
        self.mask_model = Mask_Head(320, self.yaml['nc']+1)

        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # Init weights, biases
        initialize_weights(self.model)
        self.mask_model.apply(weights_init)
        pretrained_state = torch.load(r'G:\yolov5_mask\runs\mask_rcnn_coco.pth')
        model_state = self.mask_model.state_dict()

        for k, v in pretrained_state.items():
            if 'mask' in k:
                model_key = k.replace('mask.', '')
                if model_key in model_state and v.size() == model_state[model_key].size():
                    model_state.update({model_key: v})

        self.mask_model.load_state_dict(model_state)

        # self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train


    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        x_features = []

        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if i == len(self.model) - 1:
                x_features = x.copy()

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x, x_features

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        # self.info()
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5x.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()


