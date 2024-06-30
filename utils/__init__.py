# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""utils/initialization."""

import contextlib
import platform
import threading


def emojis(str=""):
    """Returns an emoji-safe version of a string, stripped of emojis on Windows platforms."""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str


class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=""):
        """Initializes TryExcept with an optional message, used as a decorator or context manager for error handling."""
        self.msg = msg

    def __enter__(self):
        """Enter the runtime context related to this object for error handling with an optional message."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Context manager exit method that prints an error message with emojis if an exception occurred, always returns
        True.
        """
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """Decorator @threaded to run a function in a separate thread, returning the thread instance."""

    def wrapper(*args, **kwargs):
        """Runs the decorated function in a separate daemon thread and returns the thread instance."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    """
    Joins all daemon threads, optionally printing their names if verbose is True.

    Example: atexit.register(lambda: join_threads())
    """
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f"Joining thread {t.name}")
            t.join()


def notebook_init(verbose=True):
    """Initializes notebook environment by checking requirements, cleaning up, and displaying system info."""
    print("Checking setup...")

    import os
    import shutil

    from ultralytics.utils.checks import check_requirements

    from utils.general import check_font, is_colab
    from utils.torch_utils import select_device  # imports

    check_font()

    import psutil

    if check_requirements("wandb", install=False):
        os.system("pip uninstall -y wandb")  # eliminate unexpected account creation prompt with infinite hang
    if is_colab():
        shutil.rmtree("/content/sample_data", ignore_errors=True)  # remove colab /sample_data directory

    # System info
    display = None
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display

            display.clear_output()
        s = f"({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)"
    else:
        s = ""

    select_device(newline=False)
    print(emojis(f"Setup complete ✅ {s}"))
    return display
