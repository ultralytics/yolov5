from .cache_size import CacheSize
from .base import ImageCacheBase
from .disk import ImageCacheDisk
from .ram import ImageCacheRAM
from .none import ImageCacheNone


CACHE_MODES = {
    'none':     ImageCacheNone,
    'disk':     ImageCacheDisk,
    'ram':      ImageCacheRAM
}
