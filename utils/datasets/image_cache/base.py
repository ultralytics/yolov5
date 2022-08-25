from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from tqdm import tqdm

# #####################################
from .cache_size import CacheSize

# #####################################


class ImageCacheBase(ABC):

    def __init__(self):
        self.size = CacheSize()

    # #####################################

    def cache_items(self, img_pathes: List[str], nb_threads: int = None):
        with ThreadPoolExecutor(nb_threads) as pool:
            threaded = pool.map(self.cache_item, img_pathes)
            with tqdm(img_pathes) as bar:
                for _ in threaded:
                    bar.desc = "Cache[%s]" % self.size
                    bar.update()

    # #####################################

    def clear(self):
        self.size = CacheSize()

    # #####################################

    @abstractmethod
    def cache_item(self, img_path: str):
        raise NotImplemented

    # #####################################

    @abstractmethod
    def __getitem__(self, img_path: str) -> np.ndarray:
        raise NotImplemented

    # #####################################
