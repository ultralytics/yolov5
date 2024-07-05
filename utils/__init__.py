# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""utils/initialization."""

import contextlib
import platform
import threading


def emojis(str=""):
    """
    Returns an emoji-safe version of a string, stripped of emojis on Windows platforms.

    Args:
        str (str): Input string that may contain emojis.

    Returns:
        str: A version of the input string with emojis removed if the platform is Windows, otherwise the original string
            is returned unmodified.

    Notes:
        - This function is particularly useful for ensuring compatibility with systems or applications where emoji
          display might be problematic, such as certain Windows platforms.

    Examples:
        ```python
        safe_str = emojis("Hello 🌍")
        print(safe_str)  # On Windows: "Hello ", on other platforms: "Hello 🌍"
        ```
    """
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str


class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=""):
        """
        Initializes TryExcept with an optional message, providing a context manager and decorator for error handling.

        Args:
            msg (str): Optional. A message to display on exception. Default is an empty string.

        Returns:
            None

        Notes:
            The `TryExcept` class can be used as a decorator or with a context manager to encapsulate code blocks for error handling.
            It is particularly useful when you want to ensure that your code keeps running smoothly by catching and managing exceptions.

        Examples:
            Using `TryExcept` as a decorator:
            ```python
            @TryExcept(msg="Error occurred")
            def some_function():
                # Function code
            ```

            Using `TryExcept` as a context manager:
            ```python
            with TryExcept(msg="Error in context"):
                # Code block
            ```
        """
        self.msg = msg

    def __enter__(self):
        """
        Enter the runtime context related to this object for error handling with an optional message.

        Returns:
            TryExcept: The context manager instance for error handling.

        Examples:
            Usage as a context manager:
            ```python
            with TryExcept("An error occurred"):
                # Your code block here
                pass
            ```

            Usage as a decorator:
            ```python
            @TryExcept("An error occurred")
            def your_function():
                # Your code block here
                pass
            ```
        """
        pass

    def __exit__(self, exc_type, value, traceback):
        """
        Context manager exit method that handles exceptions, prints an emoji-safe error message if an exception
        occurred, and always suppresses the exception.

        Args:
            exc_type (type | None): The exception type, if an exception was raised.
            value (Exception | None): The exception instance, if an exception was raised.
            traceback (traceback | None): The traceback object, if an exception was raised.

        Returns:
            bool: Always returns True to suppress the exception.

        Examples:
            Use as a decorator:
            ```python
            @TryExcept("An error occurred")
            def function_that_might_fail():
                ...
            ```

            Use as a context manager:
            ```python
            with TryExcept("An error occurred"):
                ...
            ```
        """
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """
    Decorator @threaded to run a function in a separate thread.

    Args:
        func (Callable): The function to be executed in a separate thread.

    Returns:
        threading.Thread: The thread instance running the decorated function.

    Examples:
        ```python
        @threaded
        def some_function():
            # Your function code here

        # The function will run in a separate thread
        some_function()
        ```
    """

    def wrapper(*args, **kwargs):
        """Runs the decorated function in a separate daemon thread and returns the thread instance."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    """
    Joins all active daemon threads, optionally printing their names if verbose is True.

    Args:
        verbose (bool): If True, prints the names of the threads being joined. Default is False.

    Returns:
        None

    Example:
        ```python
        import atexit

        atexit.register(lambda: join_threads(verbose=True))
        ```
    """
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f"Joining thread {t.name}")
            t.join()


def notebook_init(verbose=True):
    """
    Initializes notebook environment by checking requirements, cleaning up, and displaying system info.

    Args:
        verbose (bool, optional): If True, displays detailed system information. Defaults to True.

    Returns:
        None

    Example:
        ```python
        from ultralytics.utils.initialization import notebook_init
        notebook_init(verbose=True)
        ```

    Note:
        - This function is particularly useful for setting up Jupyter notebook environments.
        - It ensures required packages are in place, cleans up unnecessary files, and provides system information to aid in debugging or performance monitoring.
        - Removes the Colab `/content/sample_data` directory if running in a Google Colab environment.
        - Prints setup completion along with a summary of system resources.
        - Checks and removes the `wandb` package to avoid unintended account creation prompts during setup.
    """
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
