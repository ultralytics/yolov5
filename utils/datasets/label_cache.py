import hashlib
import os
from pathlib import Path
from typing import List, Union, Optional

import numpy as np


def get_hash(paths: List[str]) -> str:
    """
    Returns a single hash value of a list of paths (files or dirs)
    """
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


class LabelCache:

    VERSION = 0.4
    VERSION_KEY = "version"
    HASH_KEY = "hash"
    RESULTS_KEY = "results"

    @staticmethod
    def load(path: Union[str, Path], hash: str) -> Optional[dict]:
        cache = LabelCache._safe_load(path=path)
        if all([
            cache,
            cache[LabelCache.VERSION_KEY] == LabelCache.VERSION,
            cache[LabelCache.HASH_KEY] == hash
        ]):
            return cache
        else:
            return None

    @staticmethod
    def save(path: Union[str, Path], hash: str, data: dict) -> None:
        pass  # TODO

    @staticmethod
    def _safe_load(path: Union[str, Path]) -> Optional[dict]:
        try:
            return np.load(path, allow_pickle=True).item()
        except:
            return None
