import cv2
import numpy as np

# #####################################
from .base import ImageCacheBase

# #####################################


class ImageCacheRAM(ImageCacheBase):
    """ Cache images in RAM
        It is mutch faster but cost RAM space.
    """

    def __init__(self):
        ImageCacheBase.__init__(self)
        self.cache_ = {}

    # #####################################

    def clear(self):
        ImageCacheBase.clear(self)
        self.cache_ = {}

    # #####################################

    def cache_item(self, img_path: str):
        frame = cv2.imread(img_path)
        self.cache_[img_path] = frame
        self.size.ram += frame.size * frame.itemsize

    # #####################################

    def __getitem__(self, img_path: str) -> np.ndarray:
        return self.cache_[img_path]

    # #####################################
