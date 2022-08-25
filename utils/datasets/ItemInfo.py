from PIL import Image, ImageOps
import os
from dataclasses import dataclass
from typing import List
from enum import IntEnum
import numpy as np, cv2
# #####################################
from ..geometry import Size, Boxes_xywh_n
from .segment import Segment
from .utils import IMG_FORMATS
# #####################################


class ItemStatus(IntEnum):
    found   =  0 # label file found & valid
    missing = -1 # label file missing
    empty   = -2 # label file found but empty
    corrupt = -3 # any error
# #####################################

@dataclass
class ItemInfo:
    img_file:       str
    lbl_file:       str
    status:         ItemStatus

    def __bool__(self):
        return (self.status != ItemStatus.corrupt)
# #####################################

@dataclass
class ValidItem(ItemInfo):
    shape:          Size
    segments:       List[Segment]
    # #####################################

    def __len__(self):              return len(self.segments)
    def __iter__(self):             return iter(self.segments)
    def __getittem__(self, idx):    return self.segments[idx]

    @property
    def labels(self):               return [s.label for s in self]
    @property
    def boxes(self):                return Boxes_xywh_n([s.box for s in self])
# #####################################

@dataclass
class InvalidItem(ItemInfo):
    err_message:    str
# #####################################

def starmap_load_item_info(args):
    """ allow load_item_info to be used with pool.imap """
    return load_item_info(*args)
# #####################################

def load_item_info(img_file, lbl_file, prefix):
    """ Load & validate one image-label pair """
    try:
        # check image validity
        verify_image(img_file)
        # correct if required
        shape = correct_jpg_rotation_exif(img_file)
        # verify & load labels
        status, segments = verify_n_load_labels(lbl_file)
        # build a valid item instance
        return ValidItem(img_file, lbl_file, status, Size(*shape), segments)

    except Exception as what:
        # build an invalid item instance
        return InvalidItem(img_file, lbl_file, ItemStatus.corrupt, 
            f'{prefix}WARNING: {img_file}: ignoring corrupt image/label: {what}')
# #####################################

def verify_image(img_file: str, min_size: int=10):
    """ Check image validity, raise an exception is anythning is wrong """
    # check that file exists and is a valid image 
    img = Image.open(img_file)
    img.verify()
    # check minimal image size
    shape = Size(*img.size)
    assert np.all(shape >= min_size), f'image size {shape} <{min_size} pixels'
    # check image format
    assert img.format.lower() in IMG_FORMATS, f'invalid image format {img.format}'
# #####################################

def correct_jpg_rotation_exif(img_file) -> bool:
    """ If image is a JPG, correct image file in-place 
        Return image's shape (width, height)
    """
    EXIF_ORIENTATION:   int = 0x0112
    ORIENTATION_NORMAL: int = 1

    img = Image.open(img_file)
    if img.format.lower() in ('jpg', 'jpeg'):
        orientation = img.getexif().get(EXIF_ORIENTATION, ORIENTATION_NORMAL)
        if orientation != ORIENTATION_NORMAL:
            img = ImageOps.exif_transpose(img)
            img.save(img_file, 'JPEG', subsampling=0, quality=100)

    return img.size
# #####################################

def verify_n_load_labels(lbl_file):
    if not os.path.isfile(lbl_file):
        status      = ItemStatus.missing
        segments    = []

    else:
        with open(lbl_file) as lbl_file:
            lines       = lbl_file.read().strip().splitlines()
            try:
                segments = [Segment.from_line(line) for line in lines]
                # TODO: duplication removal to be implemented back
                status = ItemStatus.found if len(segments) else ItemStatus.empty

            except Exception as what:
                raise Exception('parsing labels: %s' % what)
            
    return status, segments
# #####################################
