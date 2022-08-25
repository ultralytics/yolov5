from .base import ImageCacheBase
from .cache_size import CacheSize
from .disk import ImageCacheDisk
from .none import ImageCacheNone
from .ram import ImageCacheRAM

CACHE_MODES = {'none': ImageCacheNone, 'disk': ImageCacheDisk, 'ram': ImageCacheRAM}
