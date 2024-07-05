# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""General utils."""

import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.checks import check_requirements

from utils import TryExcept, emojis
from utils.downloads import curl_download, gsutil_getsize
from utils.metrics import box_iou, fitness

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv("RANK", -1))

# Settings
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
DATASETS_DIR = Path(os.getenv("YOLOv5_DATASETS_DIR", ROOT.parent / "datasets"))  # global datasets directory
AUTOINSTALL = str(os.getenv("YOLOv5_AUTOINSTALL", True)).lower() == "true"  # global auto-install mode
VERBOSE = str(os.getenv("YOLOv5_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
FONT = "Arial.ttf"  # https://ultralytics.com/assets/Arial.ttf

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["OMP_NUM_THREADS"] = "1" if platform.system() == "darwin" else str(NUM_THREADS)  # OpenMP (PyTorch and SciPy)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ["KINETO_LOG_LEVEL"] = "5"  # suppress verbose PyTorch profiler output when computing FLOPs


def is_ascii(s=""):
    """
    Checks if the input string contains only ASCII characters.

    Args:
        s (str | list | tuple | None, optional): Input value that will be converted to a string for ASCII check.
                                                Default is an empty string.

    Returns:
        bool: `True` if the input string contains only ASCII characters, otherwise `False`.

    Examples:
        ```python
        from ultralytics.utils import is_ascii

        result = is_ascii("Hello, World!")
        assert result == True

        result = is_ascii("こんにちは世界")
        assert result == False
        ```
    """
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)


def is_chinese(s="人工智能"):
    """
    Determines if a string contains any Chinese characters.

    Args:
        s (str, optional): The input string to check. Defaults to "人工智能".

    Returns:
        bool: `True` if the input string contains Chinese characters, otherwise `False`.

    Examples:
        ```python
        result = is_chinese("你好")
        print(result)  # Output: True

        result = is_chinese("Hello")
        print(result)  # Output: False
        ```

    Notes:
        This function uses a regular expression to check for any Unicode characters in the range of Chinese characters,
        which includes Chinese, Japanese, and Korean ideographs in the Unicode standard.
        See also: https://en.wikipedia.org/wiki/CJK_Unified_Ideographs for more details.
    """
    return bool(re.search("[\u4e00-\u9fff]", str(s)))


def is_colab():
    """
    Checks if the current environment is a Google Colab instance.

    Returns:
        bool: `True` if the code is running in a Google Colab environment, otherwise `False`.

    Examples:
        Determine if the script is running in Google Colab:

        ```python
        in_colab = is_colab()
        print(f"Running in Google Colab: {in_colab}")
        ```

    Notes:
        - This function checks for the presence of 'COLAB_GPU' in the system environment variables.
        - It is intended to help developers adjust their code to run efficiently and correctly within Google Colab environments.
    """
    return "google.colab" in sys.modules


def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook.

    This function determines if the script is executed within a Jupyter Notebook environment by checking the presence of
    the `IPython` kernel. It has been verified on platforms such as Google Colab, JupyterLab, Kaggle, and Paperspace.

    Returns:
        bool: True if running inside a Jupyter Notebook, False otherwise.

    Examples:
        ```python
        in_jupyter = is_jupyter()
        print(f"Running in Jupyter: {in_jupyter}")
        ```
    """
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        return get_ipython() is not None
    return False


def is_kaggle():
    """
    Checks if the current environment is a Kaggle Notebook by validating environment variables.

    Returns:
        bool: `True` if the current environment is a Kaggle Notebook, otherwise `False`.

    Examples:
        ```python
        if is_kaggle():
            print("Running in a Kaggle Notebook!")
        else:
            print("Not running in a Kaggle Notebook.")
        ```

    Notes:
        Kaggle Notebooks set specific environment variables which are validated by this function.
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_docker() -> bool:
    """
    Check if the process runs inside a Docker container.

    Returns:
        bool: True if the process is running in a Docker container, otherwise False.

    Examples:
        ```python
        is_in_docker = is_docker()
        print(f"Running in Docker: {is_in_docker}")
        ```
    """
    if Path("/.dockerenv").exists():
        return True
    try:  # check if docker is in control groups
        with open("/proc/self/cgroup") as file:
            return any("docker" in line for line in file)
    except OSError:
        return False


def is_writeable(dir, test=False):
    """
    Checks if a directory has write permissions, optionally by creating a temporary file to test write access.

    Args:
        dir (str | Path): The directory path to check for write permissions.
        test (bool): Whether to test write access by creating a temporary file. Defaults to False.

    Returns:
        bool: True if the directory is writable, otherwise False.

    Notes:
        When `test` is set to False, the function relies on `os.access()` to check write permissions, which may not work
        correctly on all operating systems, particularly Windows. If `test` is True, it performs a more reliable check
        by attempting to create and delete a temporary file within the directory.

    Examples:
        ```python
        from pathlib import Path
        print(is_writeable("/path/to/dir"))  # Check using os.access()
        print(is_writeable(Path("/path/to/dir"), test=True))  # Check by creating a temp file
        ```
    """
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / "tmp.txt"
    try:
        with open(file, "w"):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False


LOGGING_NAME = "yolov5"


def set_logging(name=LOGGING_NAME, verbose=True):
    """
    Configures the logging settings for the application.

    Args:
        name (str): The name of the logger. Defaults to 'yolov5'.
        verbose (bool): If `True`, sets the logging level to INFO; otherwise, sets it to ERROR. Defaults to True.

    Returns:
        None

    Notes:
        The logging level is set to INFO if `verbose` is True and the environment variable `RANK` is -1 or 0,
        indicating a non-distributed environment or the main process in a distributed environment. Otherwise,
        the logging level is set to ERROR.
    """
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == "Windows":
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging


def user_config_dir(dir="Ultralytics", env_var="YOLOV5_CONFIG_DIR"):
    """
    Returns the path to the user-specific configuration directory, using either an environment variable or OS-specific
    defaults.

    Args:
        dir (str): The default directory name to use if the environment variable is not set. Defaults to "Ultralytics".
        env_var (str): The environment variable name to check for the configuration directory path. Defaults to
            "YOLOV5_CONFIG_DIR".

    Returns:
        pathlib.Path: User configuration directory path.

    Notes:
        - If the directory specified by the environment variable does not exist, it is created.
        - If the environment variable is not set, the function uses OS-specific default directories:
            - Windows: "AppData/Roaming"
            - Linux: ".config"
            - macOS: "Library/Application Support"
        - In the case where OS-specific directories are not writable, it falls back to using the "/tmp" directory for
          GCP and AWS Lambda environments.
    """
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {"Windows": "AppData/Roaming", "Linux": ".config", "Darwin": "Library/Application Support"}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), "")  # OS-specific config dir
        path = (path if is_writeable(path) else Path("/tmp")) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


CONFIG_DIR = user_config_dir()  # Ultralytics settings dir


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initializes a profiling context for YOLOv5 with optional timing threshold and device specification.

        Args:
            t (float, optional): Timing threshold in seconds. Execution time below this threshold is ignored. Defaults to 0.0.
            device (torch.device, optional): Specifies the device for profiling, e.g., torch.device('cuda:0'). If None, uses the current device. Defaults to None.

        Returns:
            None
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """
        Initializes timing at the start of a profiling context block for performance measurement.

        Returns:
            Profile: Returns the current profiling context to enable the `with` statement functionality.

        Notes:
            The method sets up the start time for an operation, allowing the duration to be computed upon exiting the context.
            It is typically used in performance-critical sections where execution time needs to be logged or analyzed.

        Examples:
            ```python
            with Profile() as p:
                # code to profile
                perform_heavy_operations()
            ```
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Exits the profiling context, updating the timing duration.

        Args:
            type (type | None): Exception type if an exception is raised, otherwise None.
            value (Exception | None): Exception instance if an exception is raised, otherwise None.
            traceback (traceback | None): Traceback object if an exception is raised, otherwise None.

        Returns:
            None: This method does not return any value.

        Notes:
            This method computes the elapsed time since the context was entered and stores it in the `dt` attribute
            of the `Profile` instance. If the profiling was performed on a CUDA device, it ensures that CUDA
            operations are synchronized before measuring the time.

        Examples:
            ```python
            from ultralytics.utils import Profile

            with Profile() as p:
                # Perform operations to be profiled
                pass
            print(f'Elapsed time: {p.dt} seconds')
            ```
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Measures and returns the current time, synchronizing CUDA operations if the context is set to use a CUDA device.

        Returns:
            float: The current time in seconds, accounting for CUDA synchronization if applicable.

        Examples:
            Use `Profile.time()` to capture time before and after a block of code execution:

            ```python
            from ultralytics.utils import Profile

            profiler = Profile(device=torch.device("cuda"))
            start = profiler.time()

            # code block to profile
            end = profiler.time()
            elapsed_time = end - start
            print(f"Elapsed Time: {elapsed_time:.6f} seconds")
            ```
        """
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


class Timeout(contextlib.ContextDecorator):
    # YOLOv5 Timeout class. Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg="", suppress_timeout_errors=True):
        """
        Initializes a timeout context manager or decorator that raises a timeout error after a specified number of
        seconds.

        Args:
            seconds (int): The number of seconds before timing out.
            timeout_msg (str, optional): The message to be displayed when a timeout occurs. Defaults to an empty string.
            suppress_timeout_errors (bool, optional): Whether to suppress timeout errors. If True, timeout errors are suppressed.
                                                       Defaults to True.

        Returns:
            None

        Examples:
            ```python
            import time
            from ultralytics import Timeout

            # Example using as a context manager
            with Timeout(5, timeout_msg="Operation timed out"):
                time.sleep(10)  # This will raise a timeout error after 5 seconds

            # Example using as a decorator
            @Timeout(5, timeout_msg="Function timed out")
            def long_running_function():
                time.sleep(10)  # This will raise a timeout error after 5 seconds
            ```
        """
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        """
        Raises:
            TimeoutError: When a timeout event occurs, with an optional custom message.
        """
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        """
        Initializes the timeout mechanism, setting up the signal handler and starting the countdown.

        Returns:
            Timeout: The current Timeout instance.
        """
        if platform.system() != "Windows":  # not supported on Windows
            signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
            signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Summary line:
        Handles the exit of a timeout context, cancelling the alarm and optionally suppressing TimeoutError exceptions.
        
        Args:
        exc_type (type | None): The exception type, if any, that was raised during the execution of the context.
        exc_val (Exception | None): The exception instance raised, if any.
        exc_tb (traceback | None): The traceback object, if any, associated with the raised exception.
        
        Returns:
        None: This method does not return any value.
        
        Notes:
        - Only operational on non-Windows platforms due to limitations with the 'signal' module.
        - Cancels any active alarms if set, ensuring no lingering timeout signals after context exit.
        - If `suppress_timeout_errors` is True, it silences TimeoutError exceptions, allowing uninterrupted code flow.
        """
        if platform.system() != "Windows":
            signal.alarm(0)  # Cancel SIGALRM if it's scheduled
            if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
                return True


class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    def __init__(self, new_dir):
        """
        Initializes a context manager/decorator to temporarily change the working directory.

        Args:
            new_dir (str | Path): The target directory to switch to during the context block.

        Returns:
            None

        Notes:
            This class is utilized as a context manager or decorator to ensure that the working directory is changed to
            the specified path and automatically reverted back to the original directory upon exiting the context block,
            ensuring that the working environment remains consistent and unaffected by temporary changes.
        """
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        """
        Temporarily changes the working directory within a 'with' statement context.

        Returns:
            self (WorkingDirectory): Returns the instance of the WorkingDirectory class for context management.

        Examples:
            ```python
            with WorkingDirectory('/tmp/my_new_directory'):
                # Perform operations within the new directory context
                print(Path.cwd())  # Should output '/tmp/my_new_directory'
            # Back to the original directory context
            print(Path.cwd())  # Should output the original directory
            ```
        """
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restores the original working directory upon exiting a context.

        Args:
            exc_type (type | None): The exception type (if an exception occurred).
            exc_val (BaseException | None): The exception value (if an exception occurred).
            exc_tb (traceback | None): The traceback object (if an exception occurred).

        Returns:
            None: The method does not return any value.

        Notes:
            If an exception is raised within the context, the working directory is restored before the exception
            is propagated.

        Example:
            ```python
            from utils import WorkingDirectory

            with WorkingDirectory('/path/to/new/dir'):
                # perform operations in the new directory
                ...
            # automatically returns to the original directory after the operations
            ```
        """
        os.chdir(self.cwd)


def methods(instance):
    """
    Returns a list of method names for a given class or instance, excluding double underscore (dunder) methods.

    Args:
        instance (object): The class or instance from which to retrieve the method names.

    Returns:
        list[str]: List of method names excluding dunder methods.

    Examples:
        ```python
        class MyClass:
            def method1(self):
                pass
            def __hidden_method(self):
                pass

        instance = MyClass()
        print(methods(instance))  # Output: ['method1']
        ```

    Notes:
        Dunder methods (i.e., methods with names starting and ending with double underscores) are excluded from the result
        to provide a cleaner list of user-defined methods for the class or instance.
    """
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """
    Logs the arguments of the calling function, with options to include the filename and function name.

    Args:
        args (Optional[dict], optional): Dictionary of arguments to log. If None, the arguments are automatically
            inferred from the calling function. Defaults to None.
        show_file (bool, optional): If True, includes the filename in the log message. Defaults to True.
        show_func (bool, optional): If True, includes the function name in the log message. Defaults to False.

    Returns:
        None

    Notes:
        The function inspects the calling context to retrieve and format the arguments. It supports displaying
        the calling filename and function name based on the provided flags.

    Examples:
        ```python
        def example_function(a, b, c):
            print_args(locals())
        ```

        If called as `example_function(1, 2, 3)`, this would log something like:
        ```
        script_name: {'a': 1, 'b': 2, 'c': 3}
        ```
    """
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def init_seeds(seed=0, deterministic=False):
    """
    Initializes RNG seeds to ensure reproducibility, and enables deterministic algorithms if specified.

    Args:
        seed (int): The seed value for random number generators. Default is 0.
        deterministic (bool): If `True`, sets PyTorch to use deterministic algorithms, ensuring reproducibility at
            potential performance cost. Default is `False`.

    Returns:
        None

    See Also:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic and check_version(torch.__version__, "1.12.0"):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def intersect_dicts(da, db, exclude=()):
    """
    Returns the intersection of two dictionaries with matching keys and shapes, using the values from the first
    dictionary.

    Args:
        da (dict): The first dictionary.
        db (dict): The second dictionary.
        exclude (tuple | list): Keys to exclude from the intersection. Defaults to an empty tuple.

    Returns:
        dict: A dictionary containing the intersected keys and their corresponding values from `da`.

    Example:
        ```python
        da = {'a': 1, 'b': 2, 'c': 3}
        db = {'a': 1, 'b': 2, 'd': 4}
        result = intersect_dicts(da, db)
        # result: {'a': 1, 'b': 2}
        ```
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def get_default_args(func):
    """
    Returns a dictionary of a function's default arguments by inspecting its signature.

    Args:
        func (callable): The function whose default arguments are to be extracted.

    Returns:
        dict: A dictionary where keys are parameter names and values are the default values of the parameters.

    Example:
        ```python
        def example_function(a, b=1, c='default'):
            return a + b + c

        defaults = get_default_args(example_function)
        print(defaults)  # Output: {'b': 1, 'c': 'default'}
        ```
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_latest_run(search_dir="."):
    """
    Retrieves the path to the most recent 'last.pt' file within the specified directory, typically for resuming
    training.

    Args:
        search_dir (str): The root directory to search for 'last.pt' files. Defaults to the current directory.

    Returns:
        str: The path to the most recent 'last.pt' file if found, otherwise an empty string.

    Note:
        This function performs a recursive search within the specified directory to locate 'last.pt' files, which are
        usually checkpoints for machine learning model training.

    Examples:
        ```python
        latest_run_path = get_latest_run(search_dir="runs/train")
        if latest_run_path:
            print(f"Latest run checkpoint found at: {latest_run_path}")
        else:
            print("No 'last.pt' files found.")
        ```
    """
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def file_age(path=__file__):
    """
    Calculates and returns the age of a file in days based on its last modification time.

    Args:
        path (str | Path, optional): The path to the file for which age should be calculated. Defaults to `__file__`.

    Returns:
        float: Age of the file in days.

    Examples:
        ```python
        days_old = file_age('path/to/your/file.txt')
        print(f"The file is {days_old:.2f} days old.")
        ```

    Notes:
        This method is dependent on the system clock and file system metadata. The results may vary based on different
        systems and environments.
    """
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    """
    Returns a human-readable file modification date in 'YYYY-M-D' format, given a file path.

    Args:
        path (str): Path to the file. Defaults to the current file.

    Returns:
        str: File modification date in 'YYYY-M-D' format, based on its last modification time.

    Example:
        ```python
        last_modified = file_date("example.txt")
        print(last_modified)  # Output might be '2023-10-1'
        ```
    """
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path):
    """
    Returns the size of a file or directory in megabytes (MB).

    Args:
        path (str | Path): The file or directory path whose size is to be calculated.

    Returns:
        float: The size of the file or directory in megabytes (MB).

    Notes:
        - For directories, the function recursively sums the sizes of all contained files.
        - 1 MB is considered to be 2^20 bytes.

    Examples:
        ```python
        size = file_size("path/to/your/file_or_directory")
        print(f"Size in MB: {size:.2f}")
        ```
    """
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    else:
        return 0.0


def check_online():
    """
    Checks internet connectivity by attempting to create a connection to "1.1.1.1" on port 443, retries once if the
    first attempt fails.

    Returns:
        bool: `True` if the internet is reachable, otherwise `False`.

    Examples:
        ```python
        if check_online():
            print("Internet is reachable")
        else:
            print("Unable to reach the internet")
        ```
    """
    import socket

    def run_once():
        """Checks internet connectivity by attempting to create a connection to "1.1.1.1" on port 443."""
        try:
            socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
            return True
        except OSError:
            return False

    return run_once() or run_once()  # check twice to increase robustness to intermittent connectivity issues


def git_describe(path=ROOT):
    """
    Returns a human-readable git description of the repository at `path`, or an empty string on failure.

    Args:
        path (str | Path): The file system path to the git repository to describe. Default to the root directory.

    Returns:
        str: A string describing the current git state, including the most recent tag, the number of additional commits,
             and the abbreviated commit hash. Returns an empty string if the description fails.

    Examples:
        ```python
        desc = git_describe("/path/to/repo")
        print(desc)  # Outputs something like 'v1.0-2-gabcdef'
        ```

    Notes:
        An example output of this function is 'v5.0-5-g3e25f1e'. For more details, see: https://git-scm.com/docs/git-describe.
    """
    try:
        assert (Path(path) / ".git").is_dir()
        return check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    except Exception:
        return ""


@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo="ultralytics/yolov5", branch="master"):
    """
    Checks if YOLOv5 code is up-to-date with the specified repository branch and provides informational messages.

    Args:
        repo (str): The name of the GitHub repository to check against (default "ultralytics/yolov5").
        branch (str): The branch name to compare the local repository against (default "master").

    Returns:
        None

    Note:
        This function assumes that the local code is in a Git repository, the system is online, and Git is installed.
        Provides messages advising whether the repository is up-to-date or if an update via `git pull` is needed.

    Examples:
        ```python
        check_git_status()  # Checks against the default repository and branch
        check_git_status(repo="ultralytics/yolov8")  # Checks against a different repository
        check_git_status(branch="dev")  # Checks against a different branch
        ```
    """
    url = f"https://github.com/{repo}"
    msg = f", for updates see {url}"
    s = colorstr("github: ")  # string
    assert Path(".git").exists(), s + "skipping check (not a git repository)" + msg
    assert check_online(), s + "skipping check (offline)" + msg

    splits = re.split(pattern=r"\s", string=check_output("git remote -v", shell=True).decode())
    matches = [repo in s for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = "ultralytics"
        check_output(f"git remote add {remote} {url}", shell=True)
    check_output(f"git fetch {remote}", shell=True, timeout=5)  # git fetch
    local_branch = check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()  # checked out
    n = int(check_output(f"git rev-list {local_branch}..{remote}/{branch} --count", shell=True))  # commits behind
    if n > 0:
        pull = "git pull" if remote == "origin" else f"git pull {remote} {branch}"
        s += f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use '{pull}' or 'git clone {url}' to update."
    else:
        s += f"up to date with {url} ✅"
    LOGGER.info(s)


@WorkingDirectory(ROOT)
def check_git_info(path="."):
    """
    Checks YOLOv5 git info and returns a dictionary containing the remote URL, branch name, and commit hash.

    Args:
        path (str): Path to the git repository to inspect. Default is the current working directory (`"."`).

    Returns:
        dict: A dictionary containing the following keys:
            - `remote` (str): The URL of the git remote repository.
            - `branch` (str | None): The name of the active branch or `None` if in a detached state.
            - `commit` (str): The hash of the most recent commit.

    Raises:
        RuntimeError: If the `gitpython` library is not installed or the specified path is not a valid git repository.

    Notes:
        - Ensure `git` and `gitpython` are installed to use this function. Install `gitpython` using:
            ```python
            pip install gitpython
            ```
        - This function assumes the presence of a valid git repository at the provided `path`.

    Examples:
        ```python
        from ultralytics.utils import check_git_info

        git_info = check_git_info('/path/to/repo')
        print(git_info)
        # Output: {'remote': 'https://github.com/ultralytics/yolov5', 'branch': 'main', 'commit': 'abc123def456'}
        ```

        Output when in detached HEAD state:
        ```python
        git_info = check_git_info('/path/to/repo')
        print(git_info)
        # Output: {'remote': 'https://github.com/ultralytics/yolov5', 'branch': None, 'commit': 'abc123def456'}
        ```
    """
    check_requirements("gitpython")
    import git

    try:
        repo = git.Repo(path)
        remote = repo.remotes.origin.url.replace(".git", "")  # i.e. 'https://github.com/ultralytics/yolov5'
        commit = repo.head.commit.hexsha  # i.e. '3134699c73af83aac2a481435550b968d5792c0d'
        try:
            branch = repo.active_branch.name  # i.e. 'main'
        except TypeError:  # not on any branch
            branch = None  # i.e. 'detached HEAD' state
        return {"remote": remote, "branch": branch, "commit": commit}
    except git.exc.InvalidGitRepositoryError:  # path is not a git dir
        return {"remote": None, "branch": None, "commit": None}


def check_python(minimum="3.8.0"):
    """
    Checks if the current Python version meets the minimum required version, exiting the program if not.

    Args:
        minimum (str, optional): The minimum required Python version as a string. Defaults to "3.8.0".

    Returns:
        None

    Raises:
        SystemExit: If the current Python version is lower than the specified minimum version.

    Example:
        ```python
        check_python("3.8.0")
        ```
    Note:
        Ensure that the minimum version string follows semantic versioning (e.g., "3.8.0").

        For more information, see the [Python documentation](https://docs.python.org/3/library/sys.html#sys.version_info).
    """
    check_version(platform.python_version(), minimum, name="Python ", hard=True)


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    """
    Check if the current version meets the minimum required version, logging warnings or raising errors as specified.

    Args:
        current (str): The current version string, e.g., "1.0.0".
        minimum (str): The minimum required version string, e.g., "1.5.0".
        name (str): The name of the software being version-checked, used in logging messages. Default is "version ".
        pinned (bool): If True, require the current version to exactly match the minimum version. Default is False.
        hard (bool): If True, raises an AssertionError if the version requirement is not satisfied. Default is False.
        verbose (bool): If True, logs a warning message if the version requirement is not satisfied. Default is False.

    Returns:
        result (bool): True if the version requirement is satisfied, False otherwise.

    Examples:
        ```python
        # To check if the current version 1.0.0 meets the minimum required version 0.9.0
        check_version("1.0.0", "0.9.0", name="ModuleName")

        # To enforce that the current version must be exactly 1.0.0
        check_version("1.0.0", "1.0.0", name="ModuleName", pinned=True, hard=True)
        ```

    Notes:
        - The function compares parsed version numbers using `pkg_resources.parse_version`.
        - Set `hard=True` to enforce the version requirement strictly, raising an assertion error on failure.
        - The `pinned` parameter enables strict version matching instead of the minimum version requirement.
        - The function uses global constants and functions from the `ultralytics` library, such as `LOGGER` and `emojis`.

        See also:
        - [pkg_resources.parse_version](https://setuptools.pypa.io/en/latest/pkg_resources.html#pkg_resources.parse_version)
        - https://pypi.org/project/ultralytics
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed"  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


def check_img_size(imgsz, s=32, floor=0):
    """
    Adjusts image size to meet requirements for divisibility by a given stride.

    Args:
        imgsz (int | list | tuple): Image size(s) to be adjusted. Can be a single integer (e.g., `640`) or a list/tuple
            of integers representing dimensions (e.g., `[640, 480]`).
        s (int, optional): The stride value to which the image dimensions must be divisible. Defaults to `32`.
        floor (int, optional): A minimum value for the image dimensions to ensure they are not adjusted below this limit.
            Defaults to `0`.

    Returns:
        int | list[int]: Adjusted image size(s) ensuring divisibility by the stride. If input is an integer, returns an
            integer. If input is a list/tuple, returns a list of integers.

    Notes:
        The function will ensure that the image size(s) are multiples of the given stride `s`. This is necessary for
        compatibility with various stages of neural network processing. A warning will be logged if the input image size(s)
        are modified to meet this criteria.

    Examples:
        ```python
        # Adjusting a single image size
        new_size = check_img_size(640, s=32)

        # Adjusting multiple image sizes
        new_sizes = check_img_size([640, 480], s=32)
        ```
    """
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f"WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}")
    return new_size


def check_imshow(warn=False):
    """
    Checks environment support for image display; warns on failure if `warn=True`.

    Args:
        warn (bool, optional): If True, logs a warning message on failure to support image display. Defaults to False.

    Returns:
        bool: True if the environment supports image display, False otherwise.

    Examples:
        ```python
        if check_imshow(True):
            # Environment supports image display
            cv2.imshow('example', image)
            cv2.waitKey(0)
        else:
            # Environment does not support image display
            print('Warning: Image display not supported.')
        ```
    """
    try:
        assert not is_jupyter()
        assert not is_docker()
        cv2.imshow("test", np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    """
    Validates if a file or files have an acceptable suffix and raises an error if not.

    Args:
        file (str | list | tuple): File path(s) to be checked.
        suffix (str | list | tuple): Acceptable file suffix(es).
        msg (str): Additional message to include in the error.

    Returns:
        None

    Raises:
        AssertionError: If any file does not have an acceptable suffix.

    Examples:
        ```python
        check_suffix("model.pt", suffix=(".pt", ".pth"))
        check_suffix(["model1.pt", "model2.pth"], suffix=".pt", msg="Incorrect file type.")
        ```

    Notes:
        - This function supports checking multiple files and suffixes concurrently.
        - The function will convert any provided `suffix` to lowercase for case-insensitive matching.
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=(".yaml", ".yml")):
    """
    Searches/downloads a YAML file, verifies its suffix, and returns the file path.

    Args:
        file (str | Path): Path to the YAML file. It could be a string or a `Path` object.
        suffix (tuple, optional): Allowed file suffixes for the YAML file. Defaults to (".yaml", ".yml").

    Returns:
        Path: Path to the verified YAML file.

    Raises:
        AssertionError: If the file suffix is not in the allowed `suffix` list.

    Examples:
        ```python
        path = check_yaml("config.yaml")
        ```

    Notes:
        - The function relies on URL validation and file existence.
        - Ensures compatibility with remote URLs for YAML files.
    """
    return check_file(file, suffix)


def check_file(file, suffix=""):
    """
    Checks if a file exists, downloads it if it is a URL, and validates its suffix before returning the file path.

    Args:
        file (str): Path or URL of the file to validate.
        suffix (str | tuple[str], optional): Expected file suffix(es). Defaults to an empty string.

    Returns:
        str: Validated file path.

    Raises:
        AssertionError: If the file does not have an acceptable suffix.
        AssertionError: For invalid URL downloads.

    Refer to https://github.com/ultralytics/yolov5 for more information.

    Examples:
        ```python
        yaml_file = check_file("config.yaml")
        downloaded_file = check_file("https://example.com/model.pt")
        ```

    Notes:
        - Supports local filesystem paths and URLs.
        - Checks for HTTP/HTTPS URL downloads and ClearML dataset URLs.
        - Searches for the file within 'data', 'models', and 'utils' directories if not found in the specified path.
    """
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split("?")[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f"Found {url} locally at {file}")  # file already exists
        else:
            LOGGER.info(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f"File download failed: {url}"  # check
        return file
    elif file.startswith("clearml://"):  # ClearML Dataset ID
        assert (
            "clearml" in sys.modules
        ), "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  # search
        files = []
        for d in "data", "models", "utils":  # search directories
            files.extend(glob.glob(str(ROOT / d / "**" / file), recursive=True))  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_font(font=FONT, progress=False):
    """
    Ensures specified font exists or downloads it from Ultralytics assets, optionally displaying progress.

    Args:
        font (str | Path, optional): Path to the font file. Defaults to "Arial.ttf".
        progress (bool, optional): Whether to display download progress. Defaults to False.

    Returns:
        Path: Path to the verified or downloaded font file.

    Raises:
        RuntimeError: If the font file cannot be downloaded or verified.

    Example:
        ```python
        font_path = check_font("custom_font.ttf")
        ```
    """
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f"https://ultralytics.com/assets/{font.name}"
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_dataset(data, autodownload=True):
    """
    Validates and optionally auto-downloads a dataset configuration, returning its parsed YAML configuration as a
    dictionary.

    Args:
        data (str | Path): Path to the dataset YAML configuration file, or its name.
        autodownload (bool): Flag indicating whether to auto-download the dataset if not found. Default is True.

    Returns:
        dict: Parsed dataset configuration including paths and metadata.

    Raises:
        FileNotFoundError: If the dataset path provided does not exist and cannot be auto-downloaded.
        AssertionError: If required fields ('train', 'val', 'names') are missing in the dataset configuration.

    Example:
        ```python
        data_config = check_dataset("data/coco128.yaml")
        ```

    Note:
        This function expects the dataset YAML file to contain mandatory fields such as 'train', 'val', and 'names'. It also
        supports auto-downloading the dataset if the 'download' key is specified and `autodownload` is True.
        See [COCO128 Dataset](https://github.com/ultralytics/yolov5) for example configuration.
    """

    # Download (optional)
    extract_dir = ""
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f"{DATASETS_DIR}/{Path(data).stem}", unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob("*.yaml"))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary

    # Checks
    for k in "train", "val", "names":
        assert k in data, emojis(f"data.yaml '{k}:' field missing ❌")
    if isinstance(data["names"], (list, tuple)):  # old array format
        data["names"] = dict(enumerate(data["names"]))  # convert to dict
    assert all(isinstance(k, int) for k in data["names"].keys()), "data.yaml names keys must be integers, i.e. 2: car"
    data["nc"] = len(data["names"])

    # Resolve paths
    path = Path(extract_dir or data.get("path") or "")  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path  # download scripts
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ("train", "val", "test", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            LOGGER.info("\nDataset not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception("Dataset not found ❌")
            t = time.time()
            if s.startswith("http") and s.endswith(".zip"):  # URL
                f = Path(s).name  # filename
                LOGGER.info(f"Downloading {s} to {f}...")
                torch.hub.download_url_to_file(s, f)
                Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)  # create root
                unzip_file(f, path=DATASETS_DIR)  # unzip
                Path(f).unlink()  # remove zip
                r = None  # success
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = subprocess.run(s, shell=True)
            else:  # python script
                r = exec(s, {"yaml": data})  # return None
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf", progress=True)  # download fonts
    return data  # dictionary


def check_amp(model):
    """
    Checks PyTorch AMP functionality for a model, returns True if AMP operates correctly, otherwise False.

    Args:
        model (torch.nn.Module): The PyTorch model for which AMP functionality will be checked.

    Returns:
        bool: True if AMP operates correctly, otherwise False.

    Examples:
        ```python
        from models.common import AutoShape
        model = AutoShape(torch.hub.load('ultralytics/yolov5', 'yolov5s'))
        check_amp(model)
        ```

    Notes:
        - The function requires that the model supports AMP and is running on a CUDA device.
        - The function checks the consistency of model outputs between FP32 and AMP inference within a 10% absolute tolerance.
        - If the model fails the AMP check, a warning will be logged with a reference link to resolve potential issues.

    See also:
        - `https://github.com/ultralytics/yolov5/issues/7908`
    """
    from models.common import AutoShape, DetectMultiBackend

    def amp_allclose(model, im):
        """Compares FP32 and AMP model inference outputs, ensuring they are close within a 10% absolute tolerance."""
        m = AutoShape(model, verbose=False)  # model
        a = m(im).xywhn[0]  # FP32 inference
        m.amp = True
        b = m(im).xywhn[0]  # AMP inference
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  # close to 10% absolute tolerance

    prefix = colorstr("AMP: ")
    device = next(model.parameters()).device  # get model device
    if device.type in ("cpu", "mps"):
        return False  # AMP only used on CUDA devices
    f = ROOT / "data" / "images" / "bus.jpg"  # image to check
    im = f if f.exists() else "https://ultralytics.com/images/bus.jpg" if check_online() else np.ones((640, 640, 3))
    try:
        assert amp_allclose(deepcopy(model), im) or amp_allclose(DetectMultiBackend("yolov5n.pt", device), im)
        LOGGER.info(f"{prefix}checks passed ✅")
        return True
    except Exception:
        help_url = "https://github.com/ultralytics/yolov5/issues/7908"
        LOGGER.warning(f"{prefix}checks failed ❌, disabling Automatic Mixed Precision. See {help_url}")
        return False


def yaml_load(file="data.yaml"):
    """
    Safely loads and returns the contents of a YAML file.

    Args:
        file (str | Path): Path to the YAML file to be loaded.

    Returns:
        dict: The contents of the YAML file as a dictionary.

    Example:
        ```python
        config = yaml_load("config.yaml")
        print(config['parameter'])
        ```

    Note:
        This function uses `yaml.safe_load` for parsing the YAML content to ensure that arbitrary code execution is not
        possible.
    ```python
        return yaml.safe_load(f)
    """
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


def yaml_save(file="data.yaml", data=None):
    """
    Save a dictionary to a YAML file, converting `Path` objects to strings.

    Args:
        file (str | Path): The path to the YAML file where the data will be saved.
        data (dict): The dictionary to be saved into the YAML file.

    Returns:
        None: This function does not return any value.

    Examples:
        ```python
        data = {'key': 'value', 'path': Path('/some/path')}
        yaml_save('data.yaml', data)
        ```
    Notes:
        Path objects in the `data` dictionary are automatically converted to strings during the save operation.
    """
    if data is None:
        data = {}
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    """
    Unzips a specified ZIP file into a target directory, excluding undesired contents.

    Args:
        file (str | Path): The path to the ZIP file to be unzipped.
        path (str | Path, optional): The target directory to extract the files to. Defaults to the file's parent directory.
        exclude (tuple[str], optional): Filenames containing any of these substrings are excluded from extraction.
            Defaults to ('.DS_Store', '__MACOSX').

    Returns:
        None: This function does not return a value. It performs the unzip operation directly.

    Examples:
        ```python
        unzip_file('path/to/archive.zip', path='path/to/extract/to', exclude=('.DS_Store', 'unwanted_folder'))
        ```
    Notes:
        - The default exclude list helps filter out common unwanted files when dealing with ZIP archives created on macOS.
        - Ensure the target directory has write permissions to avoid extraction errors.
    """
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


def url2file(url):
    """
    Converts a URL string to a valid filename by stripping protocol, domain, and any query parameters.

    Args:
        url (str): The URL string that needs to be converted into a filename.

    Returns:
        str: The filename extracted from the provided URL.

    Example:
        ```python
        url = "https://example.com/path/to/file.txt?token=abc"
        filename = url2file(url)
        print(filename)   # Output: file.txt
        ```
    """
    url = str(Path(url)).replace(":/", "://")  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1, retry=3):
    """
    Downloads and optionally unzips files concurrently, supporting retries and curl fallback.

    Args:
        url (str | pathlib.Path | list[str | pathlib.Path]): URL(s) of the file(s) to download.
        dir (str | pathlib.Path, optional): Directory to save the downloaded files. Defaults to ".".
        unzip (bool, optional): If True, unzips the downloaded file(s) if they are compressed. Defaults to True.
        delete (bool, optional): If True, deletes the zip file after extracting. Defaults to True.
        curl (bool, optional): If True, uses curl for downloading. Defaults to False.
        threads (int, optional): Number of concurrent threads to use for downloading. Defaults to 1.
        retry (int, optional): Number of retry attempts for failed downloads. Defaults to 3.

    Returns:
        None

    Examples:
        ```python
        # Download a single file
        download('https://example.com/file.zip')

        # Download multiple files in parallel
        download(['https://example.com/file1.zip', 'https://example.com/file2.zip'], threads=4)
        ```
    """

    def download_one(url, dir):
        """Downloads a single file from `url` to `dir`, with retry support and optional curl fallback."""
        success = True
        if os.path.isfile(url):
            f = Path(url)  # filename
        else:  # does not exist
            f = dir / Path(url).name
            LOGGER.info(f"Downloading {url} to {f}...")
            for i in range(retry + 1):
                if curl:
                    success = curl_download(url, f, silent=(threads > 1))
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {url}...")
                else:
                    LOGGER.warning(f"❌ Failed to download {url}...")

        if unzip and success and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f"Unzipping {f}...")
            if is_zipfile(f):
                unzip_file(f, dir)  # unzip
            elif is_tarfile(f):
                subprocess.run(["tar", "xf", f, "--directory", f.parent], check=True)  # unzip
            elif f.suffix == ".gz":
                subprocess.run(["tar", "xfz", f, "--directory", f.parent], check=True)  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    """
    Adjusts `x` to be divisible by `divisor`, returning the nearest greater or equal value.

    Args:
        x (int | float): The value to be adjusted.
        divisor (int): The value by which `x` must be divisible.

    Returns:
        int: The adjusted value, which is the smallest integer greater than or equal to `x` that is divisible by `divisor`.

    Examples:
        ```python
        adjusted_value = make_divisible(64, 16)
        assert adjusted_value == 64
        adjusted_value = make_divisible(63, 16)
        assert adjusted_value == 64
        ```

    Notes:
        The function can handle both integer and floating-point inputs for `x`, ensuring the adjusted value is an integer.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    """
    Cleans a string by replacing special characters with underscores.

    Args:
        s (str): Input string to be cleaned.

    Returns:
        str: Cleaned string with special characters replaced by underscores.
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """
    Generates a lambda function that implements a sinusoidal ramp from `y1` to `y2` over a specified number of steps.

    Args:
        y1 (float, optional): The starting value of the ramp. Default is 0.0.
        y2 (float, optional): The ending value of the ramp. Default is 1.0.
        steps (int, optional): The number of steps over which the ramp is applied. Default is 100.

    Returns:
        Callable[[int], float]: A lambda function that takes an integer step index and returns the ramp value for that step.

    Example:
        ```python
        ramp = one_cycle(y1=0.1, y2=0.9, steps=50)
        value_at_step_10 = ramp(10)
        ```

    Notes:
        - The function creates a smooth, periodic transition between two values, which can be useful for learning rate
          scheduling and other gradual parameter adjustments in training machine learning models.
        - This approach is based on the "One Cycle" learning rate policy described in https://arxiv.org/pdf/1812.01187.pdf.
    """
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    """
    Colorizes a string using ANSI escape codes with specified attributes such as color and style.

    Args:
        *input (str): The string to be colorized along with optional color and style arguments. If only the string is
                    provided, defaults to "blue" and "bold".

    Returns:
        str: The colorized string with ANSI escape codes applied.

    Examples:
        ```python
        print(colorstr("green", "hello world"))  # prints "hello world" in green
        print(colorstr("red", "bold", "error"))  # prints "error" in bold red
        print(colorstr("note"))  # prints "note" in bold blue (default)
        ```
    Notes:
        For more details on ANSI escape codes, refer to https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def labels_to_class_weights(labels, nc=80):
    """
    Calculates class weights from labels to address class imbalance during training. The input labels are expected to
    have a shape of (n, 5), where 'n' is the number of labels.

    Args:
        labels (list of np.ndarray): List of label arrays, each with shape (k, 5), where 'k' is the number of annotations.
                                     Each annotation is expected to be in the format [class, x, y, width, height].
        nc (int): Number of classes. Defaults to 80.

    Returns:
        torch.Tensor: A tensor of shape (nc,) representing normalized class weights.

    Example:
        ```python
        labels = [np.array([[0, 10, 15, 20, 25], [1, 30, 35, 40, 45]]), np.array([[0, 50, 55, 60, 65]])]
        nc = 2
        class_weights = labels_to_class_weights(labels, nc)
        print(class_weights)
        ```

    Note:
        The function concatenates all label arrays, counts occurrences of each class, and then calculates the inverse
        frequency of each class as its weight. The weights are then normalized to sum to 1. Classes not represented in the
        input data will have a default weight of 1 before normalization.
    """
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """
    Calculates image weights based on class weights for weighted sampling during training.

    Args:
        labels (list): A list of numpy arrays representing labels, where each array has shape (n, 5) and first column denotes the class.
        nc (int, optional): Number of classes. Defaults to 80.
        class_weights (np.ndarray, optional): Array of class weights. Defaults to np.ones(80).

    Returns:
        np.ndarray: Image weights used for weighted sampling of images in the dataset.

    Notes:
        The returned image weights can be used to perform weighted sampling of images with:
        ```python
        index = random.choices(range(len(labels)), weights=image_weights, k=1)
        ```

    Example:
        ```python
        labels = [np.array([[0, 100, 200, 50, 50], [1, 150, 250, 60, 60]])]
        image_weights = labels_to_image_weights(labels)
        index = random.choices(range(len(labels)), weights=image_weights, k=1)
        ```

    Raises:
        ValueError: If the length of `class_weights` does not match `nc`.
        TypeError: If any label array does not have the expected shape (n, 5).

    See Also:
        - [Label-imbalance handling strategies](https://github.com/ultralytics/yolov5/issues/629)
        - [YOLOv5 training tips](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data)
    """
    # Usage: index = random.choices(range(n), weights=image_weights, k=1)  # weighted image sample
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)


def coco80_to_coco91_class():
    """
    Converts the COCO (Common Objects in Context) 80-class index to the COCO 91-class index used in the COCO paper.

    Returns:
        list[int]: A list that maps each of the 80 COCO class indices to the corresponding 91 COCO paper class indices.

    Notes:
        This function is based on the following reference:
        https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

    Example:
        ```python
        from ultralytics import coco80_to_coco91_class

        class_mapping = coco80_to_coco91_class()
        print(class_mapping[0])  # Outputs 1, the COCO 80 class index for the 'person' category
        ```
    """
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def xyxy2xywh(x):
    """
    Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, w, h] format.

    Args:
        x (torch.Tensor | np.ndarray): An array of shape (N, 4) representing N bounding boxes, where each box is defined
            by its top-left corner (x1, y1) and bottom-right corner (x2, y2).

    Returns:
        torch.Tensor | np.ndarray: An array of the same shape (N, 4) with converted bounding boxes, where each box is
        defined by its center (x, y) and dimensions (width, height).

    Example:
    ```python
    import numpy as np
    boxes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    converted_boxes = xyxy2xywh(boxes)
    ```
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] format to [x1, y1, x2, y2] format.

    Args:
        x (np.ndarray | torch.Tensor): Array of bounding boxes with shape (n, 4). Each bounding box is defined by
        [x, y, w, h], where (x, y) is the center of the box, and w and h are the width and height respectively.

    Returns:
        np.ndarray | torch.Tensor: Array of bounding boxes with shape (n, 4) in the format [x1, y1, x2, y2], where
        (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Examples:
    ```python
    import numpy as np
    import torch

    # Using numpy
    input_boxes_np = np.array([[10, 10, 4, 4]])
    output_boxes_np = xywh2xyxy(input_boxes_np)
    print(output_boxes_np)  # Output: [[ 8.  8. 12. 12.]]

    # Using torch
    input_boxes_torch = torch.tensor([[10, 10, 4, 4]])
    output_boxes_torch = xywh2xyxy(input_boxes_torch)
    print(output_boxes_torch)  # Output: tensor([[ 8.,  8., 12., 12.]])
    ```
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Converts bounding boxes from normalized [x, y, w, h] format to [x1, y1, x2, y2] format with absolute pixel values.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding boxes in normalized [x, y, w, h] format.
        w (int): The width of the image in pixels. Default is 640.
        h (int): The height of the image in pixels. Default is 640.
        padw (float): The padding width to be added to the x-coordinates. Default is 0.
        padh (float): The padding height to be added to the y-coordinates. Default is 0.

    Returns:
        np.ndarray | torch.Tensor: The converted bounding boxes in [x1, y1, x2, y2] format.

    Notes:
        - The input bounding boxes are expected to be normalized to the range [0, 1].
        - The conversion involves scaling the normalized coordinates by the image dimensions and adding the respective
          padding where applicable.

    Examples:
        ```python
        import torch
        import numpy as np

        # Example with torch tensor
        normalized_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        absolute_boxes = xywhn2xyxy(normalized_boxes, w=640, h=640)
        print(absolute_boxes)  # Output: tensor([[288., 288., 352., 352.]])

        # Example with numpy array
        normalized_boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
        absolute_boxes = xywhn2xyxy(normalized_boxes, w=640, h=640)
        print(absolute_boxes)  # Output: [[288. 288. 352. 352.]]
        ```
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding boxes from [x1, y1, x2, y2] format to normalized [x, y, w, h] format.

    Args:
        x (torch.Tensor | np.ndarray): Input bounding boxes in [x1, y1, x2, y2] format.
        w (int): Image width used for normalization. Default is 640.
        h (int): Image height used for normalization. Default is 640.
        clip (bool): If True, clip boxes to image dimensions. Default is False.
        eps (float): Small epsilon value to prevent division by zero during normalization. Default is 0.0.

    Returns:
        torch.Tensor | np.ndarray: Converted bounding boxes in normalized [x, y, w, h] format.

    Note:
        The bounding boxes [x1, y1, x2, y2] format assumes xy1=top-left, xy2=bottom-right. The normalized [x, y, w, h]
        format assumes coordinates are relative to the image dimensions.

    Example:
        ```python
        boxes = np.array([[100, 200, 300, 400], [150, 250, 350, 450]])
        normalized_boxes = xyxy2xywhn(boxes, w=640, h=640)
        ```

    Links:
        - For more details, visit the official Ultralytics repository: https://github.com/ultralytics/ultralytics
    """
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """
    Xyn2xy(x, w=640, h=640, padw=0, padh=0) Convert normalized segments into pixel segments.

    Args:
        x (np.ndarray | torch.Tensor): Input array of normalized coordinates with shape (n, 2).
        w (int, optional): Width to scale normalized x-coordinates. Default is 640.
        h (int, optional): Height to scale normalized y-coordinates. Default is 640.
        padw (int | float, optional): Horizontal padding added to the coordinates. Default is 0.
        padh (int | float, optional): Vertical padding added to the coordinates. Default is 0.

    Returns:
        np.ndarray | torch.Tensor: Output array of pixel coordinates with the same type as input `x`.

    Examples:
        ```python
        normalized_coords = np.array([[0.5, 0.5], [0.1, 0.1]])
        pixel_coords = xyn2xy(normalized_coords, w=1280, h=720)
        ```

    Notes:
        - This function is typically used to convert normalized bounding box or keypoint coordinates into pixel coordinates.
        - The input `x` should contain normalized values in the range [0, 1].
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 bounding box (xyxy) label, applying an inside-image constraint.

    Args:
        segment (numpy.ndarray | torch.Tensor): The segment points as an array of shape (n, 2), where n is the number of points.
        width (int, optional): Image width used to constrain the segment inside the image. Defaults to 640.
        height (int, optional): Image height used to constrain the segment inside the image. Defaults to 640.

    Returns:
        numpy.ndarray or torch.Tensor: A bounding box in the format [x1, y1, x2, y2]. The type of the return value matches
        the type of the input segment.

    Note:
        - The function clips the segment points to ensure they lie within the specified image dimensions.

    Example:
        ```python
        segment = np.array([[100, 50], [200, 80], [150, 200]])
        box = segment2box(segment, width=640, height=640)
        print(box)  # Output: array([100,  50, 200, 200])
        ```

        ```python
        segment = torch.tensor([[100, 50], [200, 80], [150, 200]])
        box = segment2box(segment, width=640, height=640)
        print(box)  # Output: tensor([100, 50, 200, 200])
        ```
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    (
        x,
        y,
    ) = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    """
    Converts polygon segment labels to bounding box labels.

    Args:
        segments (np.ndarray | torch.Tensor): A 3D array of shape (n, k, 2) containing n segments,
                                              each represented by k (x, y) points.

    Returns:
        np.ndarray: A 2D array of shape (n, 4) containing n bounding boxes in the format (x_min, y_min, x_max, y_max).

    Example:
        ```python
        segments = np.array([[[10, 10], [20, 10], [20, 20], [10, 20]],  # 1st segment
                             [[30, 30], [40, 30], [40, 40], [30, 40]]]) # 2nd segment
        boxes = segments2boxes(segments)
        print(boxes)
        ```

    Note:
        This utility function is commonly used to convert the polygonal annotations of object instances into rectangular
        bounding box annotations suitable for object detection models. The input `segments` can be an array or tensor.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    Resamples the input segments to a fixed number of points for consistent representation.

    Args:
        segments (list of ndarray): List of polygon segments where each segment is an ndarray of shape (n, 2), representing
                                    the coordinates of the polygon's vertices.
        n (int): Number of points to which each segment should be resampled. Default is 1000.

    Returns:
        list of ndarray: List of resampled polygon segments, each containing exactly `n` points. Each segment retains its
                         original shape as an ndarray of shape (n, 2).

    Example:
        ```python
        segments = [np.array([[0, 0], [1, 1], [2, 0]]), np.array([[1, 1], [2, 2], [3, 1]])]
        resampled_segments = resample_segments(segments, n=1000)
        ```

    Notes:
        The function ensures that the first point is repeated at the end to form a closed loop before resampling. This helps
        maintain the integrity of the polygon's shape during resampling.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Scales (xyxy) bounding boxes from img1_shape to img0_shape using optional ratio and padding parameters.

    Args:
        img1_shape (tuple): Shape of the first image as (height, width).
        boxes (np.ndarray | torch.Tensor): Bounding boxes in (xyxy) format to be scaled.
        img0_shape (tuple): Shape of the target image as (height, width).
        ratio_pad (tuple | None, optional): Ratio and padding to use for scaling. Defaults to None.

    Returns:
        boxes (np.ndarray | torch.Tensor): Scaled bounding boxes in the same format and shape as input.

    Notes:
        If `ratio_pad` is not provided, scaling and padding will be calculated based on the shapes of the images.

    Examples:
        ```python
        img1_shape = (640, 640)
        img0_shape = (1280, 720)
        boxes = np.array([[100, 100, 200, 200], [150, 150, 250, 250]])

        scaled_boxes = scale_boxes(img1_shape, boxes, img0_shape)
        ```
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    """
    Rescales segment coordinates from one image shape to another with optional normalization and padding.

    Args:
        img1_shape (tuple | list): Shape of the first image in the form (height, width).
        segments (ndarray): Segment coordinates in the form (n, 2), where n is the number of coordinates.
        img0_shape (tuple | list): Shape of the second image to which coordinates will be rescaled (height, width).
        ratio_pad (tuple | list, optional): Tuple containing the gain and padding information.
            If provided, use this ratio and padding; otherwise, they are calculated from shapes. Default is None.
        normalize (bool, optional): If True, normalize segment coordinates to the range [0, 1] based on img0_shape size.
            Default is False.

    Returns:
        ndarray: Rescaled segment coordinates array with the same shape as segments.

    Notes:
        - `img1_shape` and `img0_shape` should be provided as (height, width) tuples or lists.
        - `segments` should be an ndarray of shape (n, 2), where n is the number of coordinates.
        - Segment coordinates are rescaled according to the gain calculated from the given shapes or provided
          `ratio_pad`, and optionally normalized to fit within the [0, 1] range.

    Example:
        ```python
        img1_shape = (800, 1333)  # original image shape
        img0_shape = (640, 640)   # new image shape
        segments = np.array([[100, 200], [300, 400], [500, 600]])  # example segments
        rescaled_segments = scale_segments(img1_shape, segments, img0_shape)
        ```
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments


def clip_boxes(boxes, shape):
    """
    Clip bounding box coordinates to stay within image bounds.

    Args:
        boxes (torch.Tensor | np.ndarray): Bounding box coordinates with shape [N, 4], where N is the number of boxes and each box is represented as [x1, y1, x2, y2].
        shape (tuple[int, int]): Image dimensions in the format (height, width).

    Returns:
        None

    Notes:
        The function modifies the input 'boxes' array in place, ensuring that all bounding box coordinates stay within the specified image dimensions.

    Examples:
    ```python
    import torch

    boxes = torch.tensor([[50, 50, 150, 150], [400, 400, 500, 500], [-10, -10, 30, 30]])
    image_shape = (256, 256)
    clip_boxes(boxes, image_shape)
    # The boxes will be modified to: [[50, 50, 150, 150], [256, 256, 256, 256], [0, 0, 30, 30]]
    ```

    In this example, 'boxes' is a tensor of bounding box coordinates, and 'image_shape' specifies the dimensions of the image. The function adjusts the bounding box coordinates to ensure they lie within the image boundaries.
    ```
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(segments, shape):
    """
    Clip segment coordinates (xy1, xy2, ...) to fit within an image's boundaries given its shape (height, width).

    Args:
        segments (np.ndarray | torch.Tensor): Segment coordinates to be clipped. Shape should be (n, 2).
        shape (tuple): Tuple containing image shape in the format (height, width).

    Returns:
        None. The input segments are modified in place.

    Examples:
        ```python
        import numpy as np
        segments = np.array([[750, 650], [800, 700], [850, 750]])
        shape = (640, 640)
        clip_segments(segments, shape)
        print(segments)
        # array([[640, 640], [640, 640], [640, 640]])
        ```
    Notes:
        This function modifies the input segments array or tensor in place for performance reasons.

        When using PyTorch tensors, ensure they are on the appropriate device (CPU or CUDA) as per your use case.
    """
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Args:
        prediction (torch.Tensor): The output tensor from the model, shape [batch_size, num_boxes, 5 + num_classes + nm].
        conf_thres (float): Confidence threshold for predictions, valid values are between 0.0 and 1.0. Default is 0.25.
        iou_thres (float): Intersection over Union (IoU) threshold for NMS, valid values are between 0.0 and 1.0. Default is 0.45.
        classes (list[int] | None): List of class indices to filter by. Default is None, which does not filter by class.
        agnostic (bool): If True, class-agnostic NMS is performed. Default is False.
        multi_label (bool): If True, allows for multiple labels per box. Default is False.
        labels (list[tensor] | tuple): List of ground truth labels for each image in the batch, used for autolabelling. Default is ().
        max_det (int): Maximum number of detections per image. Default is 300.
        nm (int): Number of masks. Default is 0.

    Returns:
        list[torch.Tensor]: List of detections, one tensor per image, with shape [num_detections, 6 + nm].
                            Each tensor contains the columns [x1, y1, x2, y2, confidence, class, (optional mask columns)].

    Notes:
        - This function processes each image individually within the batch.
        - Ensure `prediction` tensor includes the required number of columns: 5 + num_classes + nm.

    Example:
    ```python
    pred = model(imgs)  # [batch_size, num_boxes, 5 + num_classes + nm]
    detections = non_max_suppression(pred, 0.25, 0.45)
    ```

    Raises:
        AssertionError: If `conf_thres` or `iou_thres` are outside the [0.0, 1.0] range, or if `prediction` tensor has incorrect dimensions.
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def strip_optimizer(f="best.pt", s=""):
    """strip_optimizer(f: str = "best.pt", s: str = ""):"""
        Strips optimizer and various training-related data from a PyTorch checkpoint, finalizing the model for deployment.
    
        Args:
            f (str): Path to the input checkpoint file. Default is 'best.pt'.
            s (str): Path to save the stripped checkpoint. If empty, the input file path is used. Default is ''.
    
        Returns:
            None
    
        Example:
            ```python
            from ultralytics import strip_optimizer
            strip_optimizer("path/to/checkpoint.pt", "path/to/stripped_checkpoint.pt")
            ```
    
        Notes:
            This function is particularly useful for reducing the file size of your trained model by removing unnecessary
            components like the optimizer state and the exponential moving average (EMA) model if present.
        """
    ```
    """
    x = torch.load(f, map_location=torch.device("cpu"))
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with ema
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr("evolve: ")):
    """
    Logs evolution results and saves them to CSV and YAML files in the specified directory, and optionally syncs the
    files with a Google Cloud Storage bucket.

    Args:
        keys (list): List of result categories.
        results (list): List of result values corresponding to the categories in `keys`.
        hyp (dict): Dictionary of hyperparameters used for evolution.
        save_dir (Path): Path to the directory for saving the CSV and YAML files.
        bucket (str): Name of the Google Cloud Storage bucket for optional synchronization.
        prefix (str, optional): Prefix for log messages. Defaults to `colorstr("evolve: ")`.

    Returns:
        None

    Notes:
        - Logs CSV and YAML files are named 'evolve.csv' and 'hyp_evolve.yaml' respectively.
        - If `bucket` is specified, the function attempts to synchronize the 'evolve.csv' file with the bucket using Google
          Cloud Storage CLI.

    Example:
        ```python
        keys = ['fitness', 'mAP']
        results = [0.593, 0.812]
        hyp = {'lr0': 0.01, 'momentum': 0.937}
        save_dir = Path('./results')
        bucket = 'my-bucket'
        print_mutation(keys, results, hyp, save_dir, bucket)
        ```
    """
    evolve_csv = save_dir / "evolve.csv"
    evolve_yaml = save_dir / "hyp_evolve.yaml"
    keys = tuple(keys) + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f"gs://{bucket}/evolve.csv"
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
            subprocess.run(["gsutil", "cp", f"{url}", f"{save_dir}"])  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = "" if evolve_csv.exists() else (("%20s," * n % keys).rstrip(",") + "\n")  # add header
    with open(evolve_csv, "a") as f:
        f.write(s + ("%20.5g," * n % vals).rstrip(",") + "\n")

    # Save yaml
    with open(evolve_yaml, "w") as f:
        data = pd.read_csv(evolve_csv, skipinitialspace=True)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :4]))  #
        generations = len(data)
        f.write(
            "# YOLOv5 Hyperparameter Evolution Results\n"
            + f"# Best generation: {i}\n"
            + f"# Last generation: {generations - 1}\n"
            + "# "
            + ", ".join(f"{x.strip():>20s}" for x in keys[:7])
            + "\n"
            + "# "
            + ", ".join(f"{x:>20.5g}" for x in data.values[i, :7])
            + "\n\n"
        )
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)

    # Print to screen
    LOGGER.info(
        prefix
        + f"{generations} generations finished, current result:\n"
        + prefix
        + ", ".join(f"{x.strip():>20s}" for x in keys)
        + "\n"
        + prefix
        + ", ".join(f"{x:20.5g}" for x in vals)
        + "\n\n"
    )

    if bucket:
        subprocess.run(["gsutil", "cp", f"{evolve_csv}", f"{evolve_yaml}", f"gs://{bucket}"])  # upload


def apply_classifier(x, model, img, im0):
    """
    Applies second-stage classifier to YOLO detection outputs, refining detections based on class predictions.

    Args:
        x (torch.Tensor): Detections from YOLO model in the format (batch, number of detections, dimensions). Each detection
          should include bounding boxes and class scores.
        model (torch.nn.Module): Second-stage classifier model that will be applied to the detections. It is expected that
          this model takes images as input and outputs class predictions.
        img (torch.Tensor): Input image tensor of the YOLO model, typically in NCHW format (batch, channels, height, width).
        im0 (np.ndarray | list): Original image(s) in numpy array format. If a batch of images is being processed, this
          should be a list of numpy arrays.

    Returns:
        None. The function modifies the input tensor `x` in-place, refining the detections by applying the classifier.

    Notes:
        - Example of a second-stage classifier: torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
        - The function adjusts the bounding boxes from YOLO detections to a square format, rescales them according to the
          original image size, and then applies the classifier for refinement.
        - The classifier input is resized to 224x224, normalized, and converted to the appropriate format before being
          applied.
        - Only detections where the YOLO class agrees with the classifier class are retained.

    Example:
        ```python
        import torch
        import torchvision.models as models
        from ultralytics.utils.general import apply_classifier

        # YOLO detections (dummy example for illustration purposes)
        detections = torch.rand((1, 10, 6))  # batch of 1, 10 detections, 6 dimensions (x1, y1, x2, y2, conf, class)

        # Load second-stage classifier (e.g., EfficientNet)
        classifier_model = models.efficientnet_b0(pretrained=True).to(device).eval()

        # Input image tensor
        img_tensor = torch.rand((1, 3, 640, 640))  # Simulation of input to YOLO model

        # Original image as numpy array
        original_image = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)

        # Apply the classifier to refine detections
        apply_classifier(detections, classifier_model, img_tensor, original_image)
        ```

    This function enhances the precision of object detection by combining the strengths of YOLO's object localization
    capabilities with a classifier's fine-grained classification ability.
    """
    # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_boxes(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for a in d:
                cutout = im0[i][int(a[1]) : int(a[3]), int(a[0]) : int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path by appending a specified separator and an integer if the path exists. Optionally
    creates directories.

    Args:
        path (str | Path): Initial path to check and increment if necessary.
        exist_ok (bool): If False (default), increments the path. If True, uses the path as-is even if it already exists.
        sep (str): Separator between the base path and the incrementing integer. Default is an empty string.
        mkdir (bool): If True, creates the incremented directory path if it does not already exist. Default is False.

    Returns:
        Path: Incremented file or directory path.

    Examples:
        ```python
        increment_path('runs/exp')          # returns 'runs/exp2' if 'runs/exp' exists
        increment_path('runs/exp', sep='_') # returns 'runs/exp_2' if 'runs/exp' exists
        increment_path('runs/exp', mkdir=True) # returns 'runs/exp2' and creates the directory if it does not exist
        ```

    Notes:
        This function is filesystem-agnostic and works across different operating systems.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------------
imshow_ = cv2.imshow  # copy to avoid recursion errors


def imread(filename, flags=cv2.IMREAD_COLOR):
    """
    Reads an image from a file and returns it as a numpy array using OpenCV's `imdecode` function to support
    multilanguage paths.

    Args:
        filename (str): The path to the image file.
        flags (int): Flags specifying the color type of a loaded image. Default is `cv2.IMREAD_COLOR`.

    Returns:
        numpy.ndarray: The loaded image in BGR format if successful, otherwise `None`.

    Examples:
        ```python
        import ultralytics.utils.general as utils
        img = utils.imread('path/to/image.jpg')
        if img is not None:
            print("Image loaded successfully")
        else:
            print("Failed to load image")
        ```

    Note:
        This function uses OpenCV's `imdecode` after reading the image as a binary stream to handle file paths with
        non-ASCII characters.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename, img):
    """Imwrite(filename: str | Path, img: np.ndarray) -> bool:"""Writes an image to a file, returns True on success and False on failure, supports multilanguage paths.
    
        Args:
            filename (str | Path): The file path where the image is to be saved. This can be a string or Path object.
            img (np.ndarray): The image matrix to be written. This is a numpy array representation of the image.
    
        Returns:
            bool: True if the image was successfully written to the file, False otherwise.
    
        Note:
            This function extends the capability of OpenCV's `imwrite` to handle file paths with non-ASCII characters by
            using `cv2.imencode` and `np.fromfile` internally.
        
        Example:
            ```python
            import numpy as np
            import cv2
            from pathlib import Path
    
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            success = imwrite(Path('/path/to/save/image.png'), img)
            print(f'Image save successful: {success}')
            ```
        """
        try:
            cv2.imencode(Path(filename).suffix, img)[1].tofile(str(filename))
            return True
        except Exception as e:
            return False
    """
    try:
        cv2.imencode(Path(filename).suffix, img)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(path, im):
    """
    Displays an image using a Unicode path with OpenCV.

    Args:
        path (str | Path): The file path of the image to display.
        im (np.ndarray): The image matrix to be displayed.

    Returns:
        None: This function does not return any value.

    Notes:
        Ensure the display environment supports OpenCV's `imshow` for correct functioning.

    Example:
        ```python
        import cv2
        from ultralytics.utils.general import imshow
        im_path = "path/to/image.jpg"
        image = cv2.imread(im_path)
        imshow(im_path, image)
        ```

    Please refer to https://github.com/ultralytics/ultralytics for detailed documentation and usage.
    """
    imshow_(path.encode("unicode_escape").decode(), im)


if Path(inspect.stack()[0].filename).parent.parent.as_posix() in inspect.stack()[-1].filename:
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow  # redefine

# Variables ------------------------------------------------------------------------------------------------------------
