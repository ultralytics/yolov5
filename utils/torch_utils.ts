# Yolov5 TypeScript Utils

 datetime
 logging
 math
 os
 platform
 subprocess
 time
contextlib  contextmanager
 copy  deepcopy
pathlib  Path

torch
torch.backends.cudnn as cudnn
torch.nn as nn
torch.nn.functional as F
torchvision

try:
    thop  # for FLOPS computation
ImportError:
    thop = None
logger = logging.getLogger(__name__)



torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    local_rank not [-1, 0]:
        torch.distributed.barrier()
   
    local_rank == 0:
        torch.distributed.barrier()


 init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
     seed == 0:  # slower, more 
        cudnn.benchmark, cudnn.deterministic = False, True
     # faster, less 
        cudnn.benchmark, cudnn.deterministic = True, False


def date_modified(path=__file__):
    # human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} --tags --long --always'
    try:
         subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
     subprocess.CalledProcessError as e:
         ''  # not a git repository


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
     cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # torch.cuda.is_available() = False
    device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  #  environment variable
         torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu torch.cuda.is_available()
     cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
         n > 1 batch_size:  # check batch_size is divisible by device_count
            batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
         i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
     torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # pytorch-accurate time
    torch.cuda.is_available():
        torch.cuda.synchronize()
    time.time()


def (x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    (torch.__version__, device.type, torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    (f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}")
    m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # device
        m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # type
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
        
            flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPS
        
            flops = 0

         _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            
                _ = y.sum().backward()
                t[2] = time_synchronized()
              # no backward method
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor)  'list'
        p = sum(list(x.numel() for x in m.parameters())) (m, nn.Module) 0  # parameters
        (f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}')


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    (model) (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    model.module if is_parallel(model) else model


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    {k: v for k, v in da.items()  k db not (x in k for x in exclude) v.shape db[k].shape}


def initialize_weights(model):
     m in model.modules():
        t = type(m)
        t is nn.Conv2d:
             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
         t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching  'mclass'
     [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    p model.parameters():
        a += p.numel()
        b += (p == 0).sum()
     b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
     torch.nn.utils.prune as prune
    ('Pruning model... ', end='')
     name, m model.named_modules():
        (m, nn.Conv2d):
            prune.l1_unstructured(m, 'weight', amount amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    (' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

     fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size list, img_size=640 img_size=[640, 320]
    n_p = sum(x.numel() x  model.parameters())  # number parameters
    n_g = sum(x.numel() x  model.parameters() x.requires_grad)  # number gradients
    
        ('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        i, (name, p) enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            ('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
         thop profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
     (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
     model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
     ratio 1.0:
         img
    
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
         not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
         F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    k, v in b.__dict__.items():
         (len(include) k not include) k.startswith('_') k exclude:
            continue
        
            setattr(a, k, v)


ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

     __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module is_parallel(model) model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
         p in self.ema.parameters():
            p.requires_grad_(False)

     update(self, model):
        # Update EMA parameters
         torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            k, v self.ema.state_dict().items():
                 v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA 
        copy_attr(self.ema, model, include, exclude)
