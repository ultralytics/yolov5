import hashlib
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional, Union, List, Dict
from abc import ABC, abstractmethod
from tqdm import tqdm

import numpy as np
import cv2

from utils.datasets.error import CacheError


def get_hash(paths: List[str]) -> str:
    """
    Returns a single hash value of a list of paths (files or dirs)
    """
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


class BaseImageCache(ABC):

    _cache_size = 0
    _loading_completed = False

    def __init__(self, cache_type: str, thread_count: int = 8) -> None:
        self._thread_count = min(thread_count, os.cpu_count())
        self._cache_type = cache_type

    @property
    def cache_size(self) -> float:
        return self._cache_size

    def load_images(self, paths: List[str]) -> None:
        self._load_images(paths=paths)
        self._loading_completed = True
        print(f"Image caching completed. ({self._cache_size / 1E9:.1f}GB {self._cache_type})")

    def get_image(self, path: str) -> np.ndarray:
        if not self._loading_completed:
            raise CacheError("Could not obtain the image. Image cache is not yet initialized.")
        image = self._get_image(path=path)
        if image is None:
            raise CacheError(f"Image with {path} path could not be found in cache.")
        return image

    @abstractmethod
    def _load_images(self, paths: List[str]) -> None:
        pass

    @abstractmethod
    def _get_image(self, path: str) -> Optional[np.ndarray]:
        pass


class DiscImageCache(BaseImageCache):

    def __init__(self, thread_count: int = 8) -> None:
        super().__init__(cache_type="disc", thread_count=thread_count)

    def _load_images(self, paths: List[str]) -> None:
        pass

    def _get_image(self, path: str) -> Optional[np.ndarray]:
        pass


class RAMImageCache(BaseImageCache):

    def __init__(self, thread_count: int = 8) -> None:
        super().__init__(cache_type="ram", thread_count=thread_count)
        self._images: Dict[str, np.ndarray] = {}

    def _load_images(self, paths: List[str]) -> None:
        results = ThreadPool(self._thread_count).imap(lambda x: self._load_image(x), paths)
        bar = tqdm(enumerate(results), total=len(paths))
        for i in bar:
            bar.desc = f"Caching images ({self._cache_size / 1E9:.1f}GB {self._cache_type})"
        bar.close()

    def _get_image(self, path: str) -> Optional[np.ndarray]:
        return self._images.get(path)

    def _load_image(self, path: str) -> None:
        image = cv2.imread(path)
        if image is None:
            raise CacheError(f"Image with {path} path could not be found.")
        self._images[path] = image
        self._cache_size += image.nbytes


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
