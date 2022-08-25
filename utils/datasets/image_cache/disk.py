from pathlib import Path

import cv2
import numpy as np

# #####################################
from .base import ImageCacheBase

# #####################################


class ImageCacheDisk(ImageCacheBase):
    """ Cache images on disk as pickle files.
        Reading from pickle files is faster than reading from original image file.
        It is faster but cost disk space.
    """
    SUFFIX = '.npy'

    def __init__(self):
        ImageCacheBase.__init__(self)
        self.cache_ = {}

    # #####################################

    def clear(self):
        ImageCacheBase.clear(self)
        self.cache_ = {}

    # #####################################

    def cache_item(self, img_path: str):
        cache_path = Path(img_path).with_suffix(self.SUFFIX)
        self.cache_[img_path] = cache_path

        if not cache_path.exists():
            frame = cv2.imread(img_path)
            np.save(cache_path.as_posix(), frame)

        self.size.disk += cache_path.stat().st_size

    # #####################################

    def __getitem__(self, img_path: str) -> np.ndarray:
        cache_path = self.cache_[img_path]
        frame = np.load(cache_path.as_posix())
        return frame

    # #####################################
