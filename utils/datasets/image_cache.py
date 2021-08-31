# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image loading and caching helpers
"""

import os
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from pathlib import Path
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
        if self._loading_completed:
            raise CacheError(f"load_images method can only be called once.")
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
        self._cache_path: Optional[str] = None

    def _load_images(self, paths: List[str]) -> None:
        self._cache_path = self._init_cache(paths=paths)
        self._image_paths = {
            path: Path(self._cache_path) / Path(path).with_suffix('.npy').name
            for path
            in paths
        }
        results = ThreadPool(self._thread_count).imap(lambda x: self._load_image(x), paths)
        bar = tqdm(enumerate(results), total=len(paths))
        for i in bar:
            bar.desc = f"Caching images ({self._cache_size / 1E9:.1f}GB {self._cache_type})"
        bar.close()

    def _get_image(self, path: str) -> Optional[np.ndarray]:
        target_path = self._image_paths.get(path)
        if target_path is None:
            return None
        return np.load(target_path)

    def _load_image(self, path: str) -> None:
        image = cv2.imread(path)
        if image is None:
            raise CacheError(f"Image with {path} path could not be found.")
        target_path = self._image_paths[path]
        np.save(target_path, image)
        self._cache_size += image.nbytes

    def _init_cache(self, paths: List[str]) -> str:
        cache_path = Path(paths[0]).parent.as_posix() + '_npy'
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        return cache_path


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
        """
        High level class responsible for loading images. ImageProvider has the ability to cache images on disk or in
        memory to speed up the loading process.

        Args:
            cache_images: `Optional[str]` - flag enabling image caching. Can be equal to one of three values: `"ram"`,
                `"disc"` or `None`. `"ram"` - all images are stored in memory to enable fastest access. This may however
                result in exceeding the limit of available memory. `"disc"` - all images are stored on hard drive but in
                raw, uncompressed form. This prevents memory overflow, and offers faster access to data then regular
                image read. `None` - image caching is turned of.
            paths: `List[str]` - list of image paths that you would like to cache.
        """
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
