from enum import Enum


class ScanState(Enum):
    """
    An enum describing the possible outcomes when scanning an individual file in the dataset.
    """
    FOUND = 1
    EMPTY = 2
    MISSING = 3
    CORRUPTED = 4
