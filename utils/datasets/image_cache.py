import os
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from typing import Optional, List, Dict

import cv2
import numpy as np
from tqdm import tqdm

from utils.datasets.error import CacheError


NUM_THREADS = min(8, os.cpu_count())


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
        self._image_paths: Dict[str, str] = {}

    def _load_images(self, paths: List[str]) -> None:
        pass  # TODO

    def _get_image(self, path: str) -> Optional[np.ndarray]:
        pass  # TODO

    def _load_image(self, path: str) -> None:
        pass  # TODO

    def _init_cache(self, paths: List[str]) -> None:
        pass  # TODO


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


class ImageProvider:

    def __init__(self, cache_images: Optional[str], paths: List[str]) -> None:
        self._cache_images = cache_images
        self._cache = ImageProvider._init_cache(cache_images=cache_images, paths=paths)

    def get_image(self, path: str) -> np.ndarray:
        if self._cache_images:
            return self._cache.get_image(path=path)
        else:
            image = cv2.imread(path)
            if image is None:
                raise CacheError(f"Image with {path} path could not be found.")
            return image

    @staticmethod
    def _init_cache(cache_images: Optional[str], paths: List[str]) -> Optional[BaseImageCache]:
        if cache_images == "disc":
            cache = DiscImageCache(thread_count=NUM_THREADS)
            cache.load_images(paths=paths)
            return cache
        if cache_images == 'ram':
            cache = RAMImageCache(thread_count=NUM_THREADS)
            cache.load_images(paths=paths)
            return cache
        if cache_images is None:
            return None
        raise CacheError(f"Unsupported cache type. Expected disc, ram or None. {cache_images} given.")
