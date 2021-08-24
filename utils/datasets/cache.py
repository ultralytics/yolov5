from pathlib import Path
from typing import Optional, Union

import numpy as np


class ImageCache:
    pass


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
    def save(path: Union[str, Path], hash: str) -> None:
        pass

    @staticmethod
    def _safe_load(path: Union[str, Path]) -> Optional[dict]:
        try:
            return np.load(path, allow_pickle=True).item()
        except:
            return None
