from .detect import run as detect
from .train import run as train
from .val import run as val

__all__ = ["train", "detect", "val"]