# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""PyTorch utils."""

import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import LOGGER, check_version, colorstr, file_date, git_describe

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")
warnings.filterwarnings("ignore", category=UserWarning)


def smart_inference_mode(torch_1_9=check_version(torch.__version__, "1.9.0")):
    """
    Applies `torch.inference_mode()` if `torch>=1.9.0`, otherwise applies `torch.no_grad()` as a decorator for functions.
    
    Args:
        torch_1_9 (bool): Indicates whether the current PyTorch version is at least 1.9.0. Default is determined
            by the `check_version` utility.
    
    Returns:
        Callable: A decorator that wraps the target function with `torch.inference_mode()` for PyTorch versions
            1.9.0 and above, otherwise with `torch.no_grad()`.
    
    Examples:
        ```python
        @smart_inference_mode()
        def inference_function(model, data):
            # Your inference code here
        ```
    """

    def decorate(fn):
        """Applies torch.inference_mode() if torch>=1.9.0, else torch.no_grad() to the decorated function."""
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate


def smartCrossEntropyLoss(label_smoothing=0.0):
    """
    Returns a CrossEntropyLoss instance with optional label smoothing, with compatibility checks for PyTorch versions.
    
    Args:
        label_smoothing (float): Value for label smoothing in the loss calculation. Default is 0.0.
    
    Returns:
        torch.nn.CrossEntropyLoss: An instance of CrossEntropyLoss with the specified label smoothing.
    
    Notes:
        - Label smoothing is supported only in PyTorch version 1.10.0 and above. If using a lower version and label
          smoothing is specified (i.e., label_smoothing > 0), a warning will be issued.
      
    Example:
        ```python
        criterion = smartCrossEntropyLoss(label_smoothing=0.1)
        ```
    """
    if check_version(torch.__version__, "1.10.0"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f"WARNING ⚠️ label smoothing {label_smoothing} requires torch>=1.10.0")
    return nn.CrossEntropyLoss()


def smart_DDP(model):
    """
    Initializes DistributedDataParallel (DDP) for model training, respecting torch version constraints.
    
    Args:
        model (torch.nn.Module): The model to be wrapped for distributed training.
    
    Returns:
        torch.nn.parallel.DistributedDataParallel: The model wrapped with DDP for distributed training.
    
    Raises:
        AssertionError: If using torch version 1.12.0 and torchvision version 0.13.0 due to a known device limitation.
    
    Notes:
        - Torch versions 1.11.0 and above support the `static_graph` parameter in DDP.
        - For torch version 1.12.0, the function raises an assertion error due to an unresolved DDP issue.
          Please refer to the issue tracker: https://github.com/ultralytics/yolov5/issues/8395 for more details.
    
    Examples:
        ```python
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP
    
        model = MyModel().to(LOCAL_RANK)
        model = smart_DDP(model)
        ```
    
    Returns:
        torch.nn.parallel.DistributedDataParallel: A DDP wrapped model for distributed training.
    """
    assert not check_version(torch.__version__, "1.12.0", pinned=True), (
        "torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. "
        "Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395"
    )
    if check_version(torch.__version__, "1.11.0"):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def reshape_classifier_output(model, n=1000):
    """
    Reshapes the last layer of the model to match a specified class count 'n', supporting various model types.
    
    Args:
        model (torch.nn.Module): The neural network model whose final layer will be reshaped. It can be a custom model
                                 class, a ResNet, EfficientNet, or any model with Linear or Conv2d layers in its Sequential 
                                 module.
        n (int): The desired number of output classes. Default is 1000.
    
    Returns:
        None: This function modifies the model in place to change the output dimension of its last layer.
    
    Raises:
        AttributeError: If the model does not have the expected module structure to reshape its final layer.
    
    Notes:
        This function currently supports the YOLOv5 Classify head, ResNet, EfficientNet, and models with Linear or Conv2d 
        layers in a Sequential container.
    
    Example:
        ```python
        from torchvision import models
    
        model = models.resnet50(pretrained=True)
        reshape_classifier_output(model, n=10)  # Change output layer to match 10 classes
        ```
    
        This will modify the final fully connected layer of ResNet50 to output predictions for 10 classes instead of the 
        default 1000 classes.
    """
    from models.common import Classify

    name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
    if isinstance(m, Classify):  # YOLOv5 Classify() head
        if m.linear.out_features != n:
            m.linear = nn.Linear(m.linear.in_features, n)
    elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
        if m.out_features != n:
            setattr(model, name, nn.Linear(m.in_features, n))
    elif isinstance(m, nn.Sequential):
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = len(types) - 1 - types[::-1].index(nn.Linear)  # last nn.Linear index
            if m[i].out_features != n:
                m[i] = nn.Linear(m[i].in_features, n)
        elif nn.Conv2d in types:
            i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # last nn.Conv2d index
            if m[i].out_channels != n:
                m[i] = nn.Conv2d(m[i].in_channels, n, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Context manager ensuring ordered operations in distributed training by making all processes wait for the leading
    process.
    
    Args:
        local_rank (int): The local rank of the process in distributed training. Expected values are -1, 0, or greater.
    
    Yields:
        None: This context manager does not yield any value.
    
    Notes:
        This function utilizes `torch.distributed.barrier` to synchronize processes in distributed training. For processes
        where `local_rank` is neither -1 nor 0, it will wait at the barrier until the leading process has reached the same
        point.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def device_count():
    """
    Returns the number of available CUDA devices; works on Linux and Windows by invoking `nvidia-smi`.
    
    Args:
        None
    
    Returns:
        int: The number of available CUDA devices detected by `nvidia-smi`.
    
    Raises:
        AssertionError: If the platform is neither Linux nor Windows.
        Exception: If there is an error in executing the `nvidia-smi` command.
    
    Examples:
        ```python
        from ultralytics.utils.torch_utils import device_count
    
        num_devices = device_count()
        print(f"Number of CUDA devices available: {num_devices}")
        ```
    """
    assert platform.system() in ("Linux", "Windows"), "device_count() only supported on Linux or Windows"
    try:
        cmd = "nvidia-smi -L | wc -l" if platform.system() == "Linux" else 'nvidia-smi -L | find /c /v ""'  # Windows
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


def select_device(device="", batch_size=0, newline=True):
    """
    Selects the appropriate computing device (CPU, CUDA GPU, MPS) for YOLOv5 model deployment and logs device information.
    
    Args:
        device (str): Desired device for computation ('cpu', 'cuda' or specific GPU IDs like '0,1,2'). Defaults to "".
        batch_size (int): Batch size for loading data. Ensures batch size is divisible by number of GPUs if using multiple 
            GPUs. Defaults to 0.
        newline (bool): Appends a newline to the logged device information. Defaults to True.
    
    Returns:
        torch.device: Selected device for computation, either CPU, CUDA, or MPS.
    
    Raises:
        AssertionError: If the specified CUDA device(s) are unavailable or an invalid CUDA device is requested, or if batch 
            size is not divisible by GPU count when using multiple GPUs.
    
    Notes:
        Logs relevant information about the platform, Python version, and PyTorch version in use, as well as details on the 
        selected computing devices including their properties.
    
    Example:
        ```python
        device = select_device(device='cuda:0', batch_size=16)
        ```
    
    Context:
        This function ensures that the selected device is available and properly configured for use, printing a summary of 
        the configuration.
    
    Links:
        - [PyTorch Elastic Run](https://pytorch.org/docs/stable/elastic/run.html)
    
    For related information about multi-GPU training, refer to Ultralytics' [documentation](https://github.com/ultralytics/yolov5).
    """
    s = f"YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


def time_sync():
    """
    Synchronizes PyTorch for accurate timing, leveraging CUDA if available, and returns the current time.
    
    Returns:
        float: The current time in seconds since the Epoch, as a floating point number, synchronized with CUDA if available.
    
    Examples:
        >>> import time
        >>> start = time_sync()
        >>> time.sleep(1)
        >>> end = time_sync()
        >>> elapsed_time = end - start
        >>> print(f"Elapsed time: {elapsed_time:.2f} seconds")
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    """
    YOLOv5 speed/memory/FLOPs profiler.
    
    Args:
      input (torch.Tensor | list): Input tensor or list of tensors to be profiled.
      ops (torch.nn.Module | list): Operation or list of operations to be profiled.
      n (int, optional): Number of iterations to execute the profiles for averaging. Defaults to 10.
      device (torch.device | str, optional): Device to run the profiling on. Defaults to None which triggers automatic device selection.
    
    Returns:
      list: List of profiling results where each result is a list containing the following elements:
        - p (int): Number of parameters in the operation.
        - flops (float): Number of floating point operations (in GFLOPs).
        - mem (float): GPU memory used (in GB).
        - tf (float): Forward pass time (in ms).
        - tb (float): Backward pass time (in ms).
        - s_in (tuple): Shape of the input tensor.
        - s_out (tuple): Shape of the output tensor.
    
    Example:
      ```python
      import torch
      import torch.nn as nn
      from ultralytics.utils.torch_utils import profile
    
      input_tensor = torch.randn(16, 3, 640, 640)
      operation1 = lambda x: x * torch.sigmoid(x)
      operation2 = nn.SiLU()
      
      profile(input_tensor, [operation1, operation2], n=100)
      ```
    """
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    print(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float("nan")
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    """
    Checks if the model is using Data Parallelism (DP) or Distributed Data Parallelism (DDP).
    
    Args:
        model (torch.nn.Module): The PyTorch model to check.
    
    Returns:
        bool: True if the model is an instance of torch.nn.DataParallel or torch.nn.parallel.DistributedDataParallel,
        False otherwise.
    
    Examples:
        ```python
        import torch
        from torch.nn import DataParallel
        from torch.nn.parallel import DistributedDataParallel as DDP
        from ultralytics import is_parallel
    
        model = ...  # assume a defined model
        model = DataParallel(model)
        print(is_parallel(model))  # True
    
        model = DDP(model)
        print(is_parallel(model))  # True
    
        model = torch.nn.Sequential(...)
        print(is_parallel(model))  # False
        ```
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    """
    Returns a single-GPU model by removing Data Parallelism (DP) or Distributed Data Parallelism (DDP) if applied.
    
    Args:
        model (torch.nn.Module): The model to be converted to a single-GPU version from DP or DDP.
    
    Returns:
        torch.nn.Module: The single-GPU model without DP or DDP encapsulation.
    
    Examples:
        ```python
        model = nn.DataParallel(model)
        model = de_parallel(model)
        ```
    
    Notes:
        - This function is useful in scenarios where a model needs to be saved or further manipulated outside of a 
          multi-GPU environment.
        - Ensure proper handling of the model state dict if switching between parallel and non-parallel modes for 
          checkpointing or deployment.
    """
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    """
    Initializes weights of specific layers in the model, including Conv2d, BatchNorm2d, and various activation functions.
    
    Args:
        model (nn.Module): The neural network model containing layers to be initialized.
    
    Returns:
        None
    
    Notes:
        - Conv2d layers are currently passed without specific initialization.
        - BatchNorm2d layers are initialized with epsilon and momentum values suitable for YOLO models.
        - Activation functions supported for initialization include Hardswish, LeakyReLU, ReLU, ReLU6, and SiLU.
    
    Examples:
        ```python
        import torch.nn as nn
        from ultralytics.yolo.utils.torch_utils import initialize_weights
    
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
    
        initialize_weights(model)
        ```
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    """
    Finds and returns the list of layer indices in the `model.module_list` that match the specified class type `mclass`.
    
    Args:
        model (torch.nn.Module): The model object containing the layers to be inspected.
        mclass (type): The class type to be matched against model layers. Default is `torch.nn.Conv2d`.
    
    Returns:
        list[int]: List of indices where the layers in `model.module_list` match the `mclass` type.
    
    Examples:
        ```python
        import torch.nn as nn
        from ultralytics import find_modules
        
        model = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 64, 5),
            nn.ReLU()
        )
        indices = find_modules(model, nn.Conv2d)
        print(indices)  # Output: [0, 2]
        ```
    
    Note:
        This function is useful when you want to selectively apply functions or modifications to specific types of layers 
        within a model, such as initializing weights or pruning.
    """
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    """
    Calculates and returns the global sparsity of a model as the ratio of zero-valued parameters to total parameters.
    
    Args:
        model (torch.nn.Module): The PyTorch model whose sparsity needs to be calculated.
    
    Returns:
        float: The global sparsity of the model, defined as the ratio of zero-valued parameters to the total number of parameters.
    
    Examples:
        ```python
        import torch.nn as nn
        from ultralytics import sparsity
    
        # Define a simple model
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
    
        # Calculate sparsity
        model_sparsity = sparsity(model)
        print(f"Model sparsity: {model_sparsity:.4f}")
        ```
    """
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    """
    Prunes Conv2d layers in a model to a specified sparsity using L1 unstructured pruning.
    
    Args:
        model (nn.Module): The neural network model containing Conv2d layers to be pruned.
        amount (float): The proportion of connections to prune (default is 0.3).
    
    Returns:
        None
    
    Notes:
        This function uses the `torch.nn.utils.prune` module to perform L1 unstructured pruning on all Conv2d layers.
        After pruning, the pruning masks are removed to make the pruning permanent.
    
    Examples:
        ```python
        from ultralytics import prune
        import torchvision.models as models
    
        model = models.resnet50(pretrained=True)
        prune(model, amount=0.5)
        ```
    """
    import torch.nn.utils.prune as prune

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    LOGGER.info(f"Model pruned to {sparsity(model):.3g} global sparsity")


def fuse_conv_and_bn(conv, bn):
    """
    Fuses Conv2d and BatchNorm2d layers into a single Conv2d layer.
    
    Args:
        conv (nn.Conv2d): Convolution layer to be fused.
        bn (nn.BatchNorm2d): BatchNorm layer to be fused.
    
    Returns:
        nn.Conv2d: Fused convolutional layer with the properties of the original Conv2d and BatchNorm2d layers.
    
    See:
        https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    
    Example:
        ```python
        fused_layer = fuse_conv_and_bn(conv_layer, bn_layer)
        model.layer = fused_layer
        ```
    
    Note:
        This function assumes that `conv` and `bn` layers are consecutively connected without any intermediate layers,
        and both layers must be part of the same model. The resulting fused layer simplifies inference by reducing the
        number of operations.
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, imgsz=640):
    """
    Prints model summary including layers, parameters, gradients, and FLOPs for a specified image size.
    
    Args:
        model (torch.nn.Module): The PyTorch model for which summary information will be printed.
        verbose (bool, optional): If True, prints detailed summary including layer names, shapes, and statistics. Default is False.
        imgsz (int | list[int], optional): The input image size to estimate FLOPs; can be an int or list of two ints. Default is 640.
    
    Returns:
        None
    
    Example:
        ```python
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model_info(model, verbose=True, imgsz=[640, 320])
        ```
    """
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1e9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f", {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs"  # 640x640 GFLOPs
    except Exception:
        fs = ""

    name = Path(model.yaml_file).stem.replace("yolov5", "YOLOv5") if hasattr(model, "yaml_file") else "Model"
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """
    Scales an image tensor by a given ratio, optionally maintaining the original shape and padding to multiples of a
    specified grid size.
    
    Args:
        img (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width).
        ratio (float, optional): Scaling ratio. Defaults to 1.0.
        same_shape (bool, optional): If True, maintains the original shape. Defaults to False.
        gs (int, optional): Grid size to which the image is padded. Defaults to 32.
    
    Returns:
        torch.Tensor: Scaled image tensor with the same or modified shape based on the `same_shape` parameter.
    
    Examples:
        ```python
        import torch
        import torch.nn.functional as F
        from utils.general import scale_img
    
        img = torch.randn(16, 3, 256, 416)  # random image tensor
        scaled_img = scale_img(img, ratio=1.5, same_shape=False, gs=32)
        ```
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """
    Copies attributes from object `b` to object `a`, with optional `include` and `exclude` filters.
    
    Args:
      a (object): The target object where attributes are being copied to.
      b (object): The source object from which attributes are copied.
      include (tuple | list, optional): Specific attributes to include during copying. Defaults to ().
      exclude (tuple | list, optional): Specific attributes to exclude during copying. Defaults to ().
    
    Returns:
      None
    
    Notes:
      - Only public attributes (those not starting with an underscore) are copied unless explicitly included.
      - If `include` is non-empty, only the attributes listed in `include` will be copied, regardless of other attributes.
    
    Example usage:
    ```python
    class A:
      pass
    
    class B:
      def __init__(self):
        self.x = 1
        self._y = 2
        self.z = 3
    
    a = A()
    b = B()
    copy_attr(a, b, include=('x', 'z'))
    print(a.x)  # Outputs: 1
    # print(a._y)  # AttributeError: 'A' object has no attribute '_y'
    print(a.z)  # Outputs: 3
    ```
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    """
    Initializes a smart optimizer for YOLOv5 with different parameter groups for weights with decay, weights without decay,
    and biases without decay.
    
    Args:
        model (torch.nn.Module): The model for which the optimizer is to be initialized.
        name (str): Name of the optimizer to use. Options are 'Adam', 'AdamW', 'RMSProp', and 'SGD'. Default is 'Adam'.
        lr (float): Learning rate for the optimizer. Default is 0.001.
        momentum (float): Momentum factor for optimizers that support it (like SGD and RMSProp). Default is 0.9.
        decay (float): Weight decay (L2 regularization) coefficient. Default is 1e-5.
    
    Returns:
        torch.optim.Optimizer: Initialized optimizer with separate parameter groups for weights with decay, weights without
        decay, and biases without decay.
    
    Raises:
        NotImplementedError: If an unknown optimizer name is passed.
    
    Notes:
        This function supports optimizers from the PyTorch library, and includes specialized configurations for commonly used 
        optimizers such as Adam and SGD.
    
    Example:
        ```python
        model = ...  # Define your model here
        optimizer = smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5)
        ```
    """
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    )
    return optimizer


def smart_hub_load(repo="ultralytics/yolov5", model="yolov5s", **kwargs):
    """
    smart_hub_load
    """
    YOLOv5 torch.hub.load() wrapper with smart error handling, adjusting torch arguments for compatibility.
    
    Args:
        repo (str): Repository from which to load the model. Default is "ultralytics/yolov5".
        model (str): Model name or path. Default is "yolov5s".
        **kwargs: Additional keyword arguments to pass to `torch.hub.load()`.
    
    Returns:
        torch.nn.Module: The loaded model.
    
    Example:
        ```python
        model = smart_hub_load(repo="ultralytics/yolov5", model="yolov5s")
        ```
    
    Notes:
        This function adds smart error handling by adjusting arguments based on the PyTorch version, particularly handling 
        GitHub API rate limit errors and new arguments required in torch 0.12 and later.
    """
    if check_version(torch.__version__, "1.9.1"):
        kwargs["skip_validation"] = True  # validation causes GitHub API rate limit errors
    if check_version(torch.__version__, "1.12.0"):
        kwargs["trust_repo"] = True  # argument required starting in torch 0.12
    try:
        return torch.hub.load(repo, model, **kwargs)
    except Exception:
        return torch.hub.load(repo, model, force_reload=True, **kwargs)


def smart_resume(ckpt, optimizer, ema=None, weights="yolov5s.pt", epochs=300, resume=True):
    """
    Resumes training from a checkpoint, updating optimizer, ema, and epochs, with optional resume verification.
    
    Args:
        ckpt (dict): Checkpoint containing model, optimizer, and training states.
        optimizer (torch.optim.Optimizer): Optimizer to be resumed.
        ema (optional): Exponential Moving Average object for model weights, default is None.
        weights (str): Path to the model weights file, default is "yolov5s.pt".
        epochs (int): Total number of epochs to train from the start, default is 300.
        resume (bool): Flag to indicate if training should be resumed, default is True.
    
    Returns:
        None
    
    Raises:
        AssertionError: If `resume` is True but no epochs are available to be resumed, indicating the training is completed.
        
    Examples:
        ```python
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        checkpoint = torch.load('checkpoint.pth')
        smart_resume(checkpoint, optimizer, ema=ema_object, weights='yolov5s.pt', epochs=300, resume=True)
        ```
    
    Notes:
        - The function assumes the checkpoint dictionary contains 'epoch', 'optimizer', 'best_fitness', and optionally 'ema'
          and 'updates'.
        - A resume will log the progress and necessary actions, ensuring the checkpoint states are applied correctly.
    """
    best_fitness = 0.0
    start_epoch = ckpt["epoch"] + 1
    if ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
        best_fitness = ckpt["best_fitness"]
    if ema and ckpt.get("ema"):
        ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
        ema.updates = ckpt["updates"]
    if resume:
        assert start_epoch > 0, (
            f"{weights} training to {epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        )
        LOGGER.info(f"Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs")
    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt["epoch"]  # finetune additional epochs
    return best_fitness, start_epoch, epochs


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        """
        Initializes simple early stopping mechanism for YOLOv5, with adjustable patience for non-improving epochs.
        
        Args:
            patience (float | int): Number of epochs to wait after fitness stops improving before stopping training. Setting
                                    this to an integer value specifies the exact number of patience epochs, while setting it to
                                    float("inf") disables early stopping. Defaults to 30.
        
        Returns:
            None (this constructor does not return value)
        
        Notes:
            - This early stopping mechanism helps in terminating training when there's no improvement in model
              performance (as measured by fitness, typically mAP) for a specified number of epochs.
            - It is useful for preventing overfitting and saving computational resources during model training.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Evaluates if training should stop based on fitness improvement and patience.
        
        Args:
            epoch (int): The current training epoch.
            fitness (float): The current fitness value, typically a model performance metric like mAP.
        
        Returns:
            bool: True if the training should stop, False otherwise.
        
        Notes:
            The early stopping mechanism tracks the highest fitness achieved and the epoch it was achieved at. If the
            current epoch does not show improvement in fitness over a specified patience interval, training is flagged
            to stop. The mechanism also logs a message when training stops, noting the best epoch and saving the best model.
        
        Example:
            ```python
            early_stopping = EarlyStopping(patience=30)
            should_stop = early_stopping(epoch=10, fitness=0.85)
            ```
        
            This evaluates if training should stop at epoch 10 with a fitness score of 0.85.培训应在第10阶段停止，适合得分为0.85。
        """
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping."
            )
        return stop


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """
        Initializes the Exponential Moving Average (EMA) model with specified decay parameters and sets the model to evaluation mode.
        
        Args:
            model (torch.nn.Module): The model to apply EMA on.
            decay (float): The decay rate for the EMA. It dictates how much of the model's parameters are retained with each update.
            tau (int): The tau value to adjust the effective decay during initial updates, ensuring more stability.
            updates (int): Initial number of updates applied to the EMA model. Useful for resuming training from a checkpoint.
        
        Returns:
            None
        Notes:
            For more details on Exponential Moving Average (EMA), refer to 
            https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        
            Example usage of initializing ModelEMA:
            ```python
            from ultralytics import ModelEMA, YOLO
            
            model = YOLO('yolov5s.pt').model
            ema = ModelEMA(model)
            ```
        
            This is useful for maintaining a smoothed version of the model weights which often improves model performance during training.
        """
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        Updates the Exponential Moving Average (EMA) parameters based on the current model's parameters.
        
        Args:
            model (nn.Module): The current model whose parameters will be used to update the EMA.
        
        Returns:
            None
        
        Notes:
            The EMA decay is dynamically computed using the formula `decay * (1 - exp(-updates / tau))` where
            `decay` is the base decay rate, `tau` is a constant to help during early epochs, and `updates`
            represents the number of times the EMA has been updated. Only floating-point parameters are updated.
        """
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """
        Updates EMA attributes by copying specified attributes from the model to the EMA.
        
        Args:
            model (object): The source model from which attributes are copied to the EMA.
            include (tuple): Tuple of attribute names to include when copying. Default is empty tuple, meaning all attributes
                are included.
            exclude (tuple): Tuple of attribute names to exclude when copying. Defaults to ("process_group", "reducer").
        
        Returns:
            None
        
        Note:
            This function selectively copies attributes from the given model to the internal EMA model. Attributes included in 
            the `include` list will always be copied, provided they are not also included in `exclude`. The function is designed 
            to handle specific exclusions by default, making it useful for updating model states while avoiding certain 
            attributes that may interfere with the EMA's functioning.
            
        Example:
            ```python
            ema = ModelEMA(model)
            ema.update_attr(model, include=('attr1', 'attr2'), exclude=('reducer',))
            ```
        """
        copy_attr(self.ema, model, include, exclude)
