import glob
import hashlib
import json
import os
import pickle
import random
from collections import Counter
from importlib import import_module
from itertools import chain, repeat
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import schema as S
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import LOGGER, NUM_THREADS

from ..geometry import Boxes_xywh_n, Size
from .image_cache import CACHE_MODES, ImageCacheBase
from .inl_data import InL_Data
# #####################################
from .ItemInfo import ItemStatus, starmap_load_item_info
from .utils import BAR_FORMAT, HELP_URL, IMG_FORMATS, img2label_paths

# #####################################


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.71  # dataset labels *.cache version

    def __init__(
            self,
            path: Union[str, List[str]],
            img_size: Union[int, Tuple[int, int]] = 640,
            hyp: Mapping[str, Any] = None,
            single_cls: bool = False,
            stride: int = 32,
            prefix: str = '',
            cache_mode: str = 'none',
            ignore_cache: bool = False,
            augment: bool = False,
            rect: bool = False,
            batch_size: int = 16,
            **kwargs  # allow skipping obsolete parameters
    ):
        """
            path:           a folder or a list of folders
            img_size:       (w, h) or max_size
            hyp:            hyperparameters dict
            image_weights:  ?
            cache_images:   ?
            single_cls:     all labels are merged together as '0'
            stride:         output must be a multiple of stride
            prefix:         prefix used for logging
            ignore_cache:   ignore cache even if one available
            augment:        Enable Augment Transforms & Mosaic
            rect:           Use batch's frame max size (ignore img_size)
                            deactivate augment (thus mosaic)
                            incompatible with dataloader shuffle
                            sort items by shape
            batch_size:     nb items in the batch -> used when rect activated
        """
        self.path = path
        self.hyp = hyp
        self.augment = augment and not rect
        self.rect = rect
        self.batch_size = batch_size

        # ensure img_size is (w, h)
        if isinstance(img_size, int):
            img_size = Size(img_size, img_size)
        else:
            img_size = Size(*img_size)
        # ensure img_size is a multiple of stride
        self.img_size = self.match_stride(img_size, stride)

        # load image files
        img_files, p = self.list_img_files(path, prefix)
        # load label files
        lbl_files = img2label_paths(img_files)
        # build cache path
        cache_path = self.build_cache_path(p, lbl_files)
        # build cache hash
        cache_hash = self.compute_hash(img_files, lbl_files)
        # try to load from cache
        if not ignore_cache:
            data = self.load_cache(cache_path, cache_hash, prefix)
        else:
            data = None
        # if not found nor valid
        if data is None:
            # load from source files
            data = self.load_from_sources(img_files, lbl_files, prefix)
            # save new cache
            self.save_cache(cache_path, cache_hash, data, prefix)
        # print loaded data information
        LOGGER.info(f'{prefix} Loaded %d files: %s' % (data.n_files, self.stats_to_str(data.stats)))
        LOGGER.info(f'{prefix} Found %d images' % data.n_images)
        LOGGER.info(f'{prefix} Found %d labels: %s' % (
            data.n_labels, 
            ', '.join('%d:%d' % (lbl, n) for lbl, n in data.label_stats.items())
        ))
        if len(data.invalid_items):
            LOGGER.info('\n'.join(map(lambda item: item.err_message, data.invalid_items)))
        # ensure any data available
        assert data.n_labels, \
            f'{prefix}No labels loaded. Can not train without labels. See {HELP_URL}'
        # Update labels in-place
        self.update_labels(data, single_cls)
        # manage rect
        if self.rect:
            self.batch_shapes = self.prepare_for_rect(data, batch_size, img_size, stride)
        else:
            self.batch_shapes = None
        # store internaly
        self.data: InL_Data = data
        # get labels
        self.labels = self.data.labels
        # get shapes
        self.shapes = np.vstack(self.data.shapes)
        # build augment transforms from hyp config
        if self.augment:
            self.augment_transforms = self.build_augment_transforms(hyp)
        else:
            self.augment_transforms = []

        # Cache images into RAM/disk for faster training
        # (WARNING: large datasets may exceed system resources)
        CacheClass = CACHE_MODES.get(cache_mode.lower())
        assert CacheClass is not None, 'Unknown cache_mode: %s' % cache_mode
        self.cache: ImageCacheBase = CacheClass()
        self.cache.cache_items([item.img_file for item in self.data], NUM_THREADS)

    # #####################################

    @staticmethod
    def match_stride(size: Size, stride: int) -> Size:
        """ returns the first multiple of stride >= size """
        return (np.ceil(size / stride) * stride).astype(int)

    # #####################################

    @classmethod
    def prepare_for_rect(cls, data: InL_Data, batch_size: int, img_size: Size, stride: int) -> List[Size]:
        """ pre-compute per-batch expected image size
            when using rect
                dataloader shuffle is False
                dataloader drop_last is False
        """
        # get batch count (drop_last=False)
        n_batch = int(np.ceil(len(data) / batch_size))
        # sort items by aspect ratio
        data.valid_items.sort(key=lambda item: item.shape.ar)
        # get per-item aspect ratio with NAN padding on last batch
        AR = [item.shape.ar for item in data]
        n_pad = (n_batch * batch_size) - len(data)
        AR = np.array(AR + [np.nan] * n_pad)
        # compute per-bach min & max aspect ratios
        BAR = AR.reshape((-1, batch_size))
        bar_min = np.nanmin(BAR, 1)
        bar_max = np.nanmax(BAR, 1)

        # if bar_max < 1.0 -> [1, bar_max]
        # if bar_min > 1.0 -> [1 / bar_min, 1]
        # else             -> [1, 1]
        shapes = np.array([
            (1.0, ar_max) if (ar_max < 1.0) else 
            (1.0 / ar_min, 1.0) if (ar_min > 1.0) else 
            (1.0, 1.0)
            for ar_min, ar_max
            in zip(bar_min, bar_max)
        ])
        shapes = cls.match_stride(shapes * img_size, stride)

        return shapes

    # #####################################

    @staticmethod
    def list_img_files(path: Union[str, List[str]], prefix: str) -> List[Path]:
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception(f'{prefix}{p} does not exist')

            img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        return img_files, p

    # #####################################

    @staticmethod
    def build_cache_path(p, lbl_files):
        return (p if p.is_file() else Path(lbl_files[0]).parent).with_suffix('.cache')

    # #####################################

    @classmethod
    def load_cache(cls, cache_path: str, cache_hash: bytes, prefix: str):
        try:
            assert os.path.exists(cache_path), 'cache file not found: %s' % cache_path
            with open(cache_path, 'rb') as cache_file:
                # read & check stored hash
                stored_hash = cache_file.read(len(cache_hash))
                assert stored_hash == cache_hash, 'incompatible cache file found: %s' % cache_path
                # read content
                cache_data = pickle.load(cache_file)
                LOGGER.info(f'{prefix} loaded from cache file')
                return cache_data

        except Exception as what:
            LOGGER.info(f'{prefix} {what}')
            return None

    # #####################################

    @staticmethod
    def save_cache(cache_path: str, cache_hash: bytes, data, prefix: str):
        # open the file in binary mode
        with open(cache_path, 'wb') as cache_file:
            # write the hash first
            cache_file.write(cache_hash)
            # happend serialized data
            pickle.dump(data, cache_file)
        LOGGER.info(f'{prefix} new cache saved')

    # #####################################

    @classmethod
    def load_from_sources(cls, img_files: List[str], lbl_files: List[str], prefix: str) -> InL_Data:
        desc = f"{prefix}Scanning images and labels: "
        stats = Counter({status: 0 for status in ItemStatus})
        valid_items = []
        invalid_items = []

        with Pool(NUM_THREADS) as pool:
            items = pool.imap(starmap_load_item_info, zip(img_files, lbl_files, repeat(prefix)))
            pbar = tqdm(items, desc=desc, total=len(img_files), bar_format=BAR_FORMAT, leave=False)
            for item_info in pbar:
                stats.update([item_info.status])
                if item_info:
                    valid_items.append(item_info)
                else:
                    invalid_items.append(item_info)

                pbar.desc = f"{desc}" + cls.stats_to_str(stats)
            pbar.close()

        # compute label stats
        label_stats = Counter(chain(*map(lambda i: i.labels, valid_items)))

        return InL_Data(stats, label_stats, valid_items, invalid_items)

    # #####################################

    @staticmethod
    def stats_to_str(stats: Counter) -> str:
        return ", ".join(["%d %s" % (stats[status], status.name) for status in ItemStatus])

    # #####################################

    @classmethod
    def compute_hash(cls, img_files, lbl_files):
        size_if_exist = lambda p: os.path.getsize(p) if os.path.exists(p) else None
        hash_content = {
            'version': cls.cache_version,
            'items':
            [(img_file, lbl_file, size_if_exist(lbl_file)) for img_file, lbl_file in zip(img_files, lbl_files)]}
        hash_buffer = json.dumps(hash_content).encode()
        md5 = hashlib.md5()
        md5.update(hash_buffer)
        return md5.digest()

    # #####################################

    @staticmethod
    def update_labels(data: InL_Data, single_cls: bool):
        # force all segments to have a signle label : 0
        if single_cls:
            for item in data:
                for segment in item:
                    segment.label = 0

    # #####################################

    @staticmethod
    def build_sizing_transforms(src_size: Size,
                                dst_size: Size,
                                allow_scaleup: bool,
                                allow_scaledown: bool,
                                fill_color=(114, 114, 144)):

        transforms = []

        # resize preserving aspect ratio
        ratio = (dst_size / src_size).min()
        if not allow_scaleup:
            ratio = min(ratio, 1.0)
        if not allow_scaledown:
            ratio = max(ratio, 1.0)
        new_size = (src_size * ratio).astype(int)
        Transform = A.Resize(new_size.h, new_size.w, always_apply=True)
        transforms.append(Transform)

        # compute paddings
        padding = (dst_size - new_size).clip(0)

        # padding if required
        Transform = A.PadIfNeeded(dst_size.h,
                                  dst_size.w,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  value=fill_color,
                                  position=A.PadIfNeeded.PositionType.CENTER,
                                  always_apply=True)
        transforms.append(Transform)

        # cropping if required
        Transform = A.CenterCrop(dst_size.h, dst_size.w, always_apply=True)
        transforms.append(Transform)

        return transforms, padding

    # #####################################

    @classmethod
    def build_augment_transforms(cls, hyp):
        """ see https://albumentations.ai/docs/api_reference/full_reference/
            for available Albumentations transforms
            build transform instances from YAML file (hyp.yml)
            Transforms MUST NOT change crop size !
        """
        # build transforms from hyp config
        schema = S.Schema(
            {
                S.Optional('min_area', default=0.0):
                float,
                S.Optional('min_visibility', default=0.0):
                float,
                'transforms': [{
                    'name': str,
                    S.Optional('module', default='albumentations'): str,
                    S.Optional('p', default=1.0): float,
                    S.Optional('kwargs', default={}): dict}]},
            ignore_extra_keys=True)
        transform = schema.validate(hyp['transform'])
        transforms = []

        # add all transforms declared in the hyp config file
        for trans in transform['transforms']:
            if trans['p'] > 0.0:
                # load transform's python module (by default albumentations)
                try:
                    module = import_module(trans['module'])
                except Exception:
                    raise Exception('Transform Module Not Found: %s' % trans['module'])
                # load transform from this module
                try:
                    Transform = getattr(module, trans['name'])
                except Exception:
                    raise Exception('Unknown Transform: "{}" @ {}'.format(trans['name'], trans['module']))
                # try to use provided kwargs
                try:
                    Transform = Transform(**trans['kwargs'], p=trans['p'])
                except Exception:
                    raise Exception('Invalid kwargs for transform {} @ {} : {!r}'.format(
                        trans['name'], trans['module'], trans['kwargs']))
                # add the transform to the list
                transforms.append(Transform)

        return transforms

    # #####################################

    def use_mosaic(self):
        """ decide to use mosaic for one sample based on hyp """
        proba = self.hyp.get('transform', {}).get('mosaic', 0.0)
        return np.random.random() < proba

    # #####################################

    def use_mixup(self):
        """ decide to use mixup for one sample based on hyp """
        proba = self.hyp.get('transform', {}).get('mixup', 0.0)
        return np.random.random() < proba

    # #####################################

    def use_copy_paste(self):
        """ decide to use copy_paste for one sample based on hyp """
        proba = self.hyp.get('transform', {}).get('copy_paste', 0.0)
        return np.random.random() < proba

    # #####################################

    def __len__(self):
        return len(self.data)

    # #####################################

    def load_item(self, index: int):
        item = self.data.valid_items[index]
        frame = self.cache[item.img_file]
        assert frame is not None, 'Img file not found or invalid'
        labels = item.labels
        boxes = item.boxes
        return frame, labels, boxes

    # #####################################

    def __getitem__(self, index):
        # get img size to use
        if not self.rect:
            img_size = self.img_size
        else:
            img_size = self.batch_shapes[index // self.batch_size]

        # get frame, labels & boxes
        if self.augment and self.use_mosaic():
            # mosaic only applicable when augment allowed
            frame, labels, boxes = self.load_mosaic(index)
        else:
            frame, labels, boxes = self.load_item(index)

        # apply special transforms not compatible with Albumentations
        # both image and boxes are modified together
        if self.augment:
            # if required, apply copy_paste transform
            if self.use_copy_paste():
                raise NotImplementedError  # TODO
            # if required, apply mixup transform
            if self.use_mixup():
                raise NotImplementedError  # TODO

        # ensure that boxes are on [0;1]
        boxes = boxes.to_xyxy_n().clip(0, 1).to_xywh_n()
        # keep boxes that are not empty
        valid = boxes.A > 0
        boxes = boxes[valid]
        labels = np.array(labels)[valid]

        # get original shape
        ho, wo = frame.shape[:2]

        # build albumentations transform
        sizing_tranforms, padding = self.build_sizing_transforms(src_size=Size(wo, ho),
                                                                 dst_size=img_size,
                                                                 allow_scaledown=True,
                                                                 allow_scaleup=self.augment)
        Transform = A.Compose(
            transforms=[
                *sizing_tranforms,
                *self.augment_transforms,  # empty list if not self.augment
                # numpy frame (h, w, c) -> torch tensor (c, h, w)
                ToTensorV2(always_apply=True)],
            bbox_params=A.BboxParams(
                format='yolo',  # xywh_n
                min_area=self.hyp['transform'].get('min_area', 0.0),
                min_visibility=self.hyp['transform'].get('min_visibility', 0.0),
                label_fields=['labels'],
                check_each_transform=False))

        # prepare for albumentations
        transformed = Transform(image=frame, bboxes=boxes, labels=labels)
        # retrieve frame (is a proper Tensor)
        frame = transformed['image']

        # for COCO mAP rescaling
        hr, wr = frame.shape[-2:]
        shapes = (ho, wo), ((hr / ho, wr / wo), padding)

        # combine labels & boxes [(l, cx, cy, w, h), ...]
        labels = transformed['labels']
        boxes = Boxes_xywh_n(transformed['bboxes'])

        labels_yolo = torch.zeros((len(labels), 6))
        labels_yolo[:, 1] = torch.Tensor(labels)
        labels_yolo[:, 2:] = torch.from_numpy(boxes)

        img_path = self.data.valid_items[index].img_file

        return frame, labels_yolo, img_path, shapes

    # #####################################

    def load_mosaic(self, src_index: int, fill_color=(114, 144, 144)):
        """ build a 2x2 mosaic image
            src_index is used as top-left tile
            other tiles are picked randomly
            each tile is padded when required to stack.
            padding is made so that tiles are as centered as possible
        """
        population = set(range(len(self.data))) - {src_index}
        indices = random.sample(population, k=3)
        indices = [src_index, *indices]
        items = [self.data.valid_items[indice] for indice in indices]
        shapes = np.array([item.shape for item in items])
        lw, rw = shapes[:, 0].reshape((2, 2)).max(0)
        th, bh = shapes[:, 1].reshape((2, 2)).max(1)
        w, h = lw + rw, th + bh
        tl, tr, bl, br = items

        # Top-Left
        pad_x = lw - tl.shape.w
        pad_y = th - tl.shape.h
        tl_frame = cv2.copyMakeBorder(self.cache[tl.img_file],
                                      pad_y,
                                      0,
                                      pad_x,
                                      0,
                                      cv2.BORDER_CONSTANT,
                                      value=fill_color)
        tl_labels = tl.labels
        tl_boxes = (tl.boxes.to_xywh(*tl.shape) + (pad_x, pad_y, 0, 0)).to_xywh_n(w, h)

        # Top-Right
        pad_x = rw - tr.shape.w
        pad_y = th - tr.shape.h
        tr_frame = cv2.copyMakeBorder(self.cache[tr.img_file],
                                      pad_y,
                                      0,
                                      0,
                                      pad_x,
                                      cv2.BORDER_CONSTANT,
                                      value=fill_color)
        tr_labels = tr.labels
        tr_boxes = (tr.boxes.to_xywh(*tr.shape) + (lw, pad_y, 0, 0)).to_xywh_n(w, h)

        # Bottom-Left
        pad_x = lw - bl.shape.w
        pad_y = bh - bl.shape.h
        bl_frame = cv2.copyMakeBorder(self.cache[bl.img_file],
                                      0,
                                      pad_y,
                                      pad_x,
                                      0,
                                      cv2.BORDER_CONSTANT,
                                      value=fill_color)
        bl_labels = bl.labels
        bl_boxes = (bl.boxes.to_xywh(*bl.shape) + (pad_x, th, 0, 0)).to_xywh_n(w, h)

        # Bottom-Right
        pad_x = rw - br.shape.w
        pad_y = bh - br.shape.h
        br_frame = cv2.copyMakeBorder(self.cache[br.img_file],
                                      0,
                                      pad_y,
                                      0,
                                      pad_x,
                                      cv2.BORDER_CONSTANT,
                                      value=fill_color)
        br_labels = br.labels
        br_boxes = (br.boxes.to_xywh(*br.shape) + (lw, th, 0, 0)).to_xywh_n(w, h)

        # combine
        frame = np.vstack([
            np.hstack([tl_frame, tr_frame]),
            np.hstack([bl_frame, br_frame]),])
        labels = tl_labels + tr_labels + bl_labels + br_labels
        boxes = np.vstack([tl_boxes, tr_boxes, bl_boxes, br_boxes]).view(Boxes_xywh_n)

        return frame, labels, boxes

    # #####################################
