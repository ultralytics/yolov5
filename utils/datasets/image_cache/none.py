import numpy as np, cv2
# #####################################
from .base import ImageCacheBase
# #####################################

class ImageCacheNone(ImageCacheBase):
    """ No cache, images are loaded from original image files 
        It is slower but cost no additionnal ROM/RAM
    """
    def cache_item(self, img_path: str):
        pass # nothing to do
    # #####################################

    def __getitem__(self, img_path: str) -> np.ndarray:
        return cv2.imread(img_path)
    # #####################################
