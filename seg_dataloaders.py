## TODO: Move to utils, merge with dataloaders.py

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders
"""

import json
import logging
import time
import numpy as np
from functools import wraps
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import distributed
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import RandomSampler

from seg_augmentations import (Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective, )
from utils.general import colorstr, check_dataset, check_yaml, xywhn2xyxy, xyxy2xywhn, xyn2xy
from utils.torch_utils import torch_distributed_zero_first


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, augment=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.augment, idx) for idx in batch]


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
        rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix="", shuffle=False, 
        area_thr=0.2, mask_head=False, mask_downsample_ratio=1, overlap_mask=False):
    if rect and shuffle:
        print("WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    data_load = LoadImagesAndLabelsAndMasks if mask_head else LoadImagesAndLabels
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = data_load(path, imgsz, batch_size, augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache, single_cls=single_cls, stride=int(stride), pad=pad, image_weights=image_weights,
            prefix=prefix, area_thr=area_thr, )
        if mask_head:
            dataset.downsample_ratio = mask_downsample_ratio
            dataset.overlap = overlap_mask

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # sampler = InfiniteSampler(len(dataset), seed=0)
    sampler = (distributed.DistributedSampler(dataset, shuffle=shuffle) if rank != -1 else RandomSampler(dataset))

    batch_sampler = (YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False,
        augment=augment, ) if not rect else None)
    dataloader = DataLoader(dataset, num_workers=nw, batch_size=1 if batch_sampler is not None else batch_size,
        # batch-size and batch-sampler is exclusion
        batch_sampler=batch_sampler, pin_memory=True,
        collate_fn=data_load.collate_fn4 if quad else data_load.collate_fn,
    )
    return dataloader, dataset


class Dataset(torchDataset):
    """This class is a subclass of the base :class:`torch.utils.data.Dataset`,
    that enables on the fly resizing of the ``input_dim``.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
    """

    def __init__(self, augment=True):
        super().__init__()
        self.augment = augment

    @staticmethod
    def mosaic_getitem(getitem_fn):
        """
        Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the closing mosaic

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.mosaic_getitem
            ...     def __getitem__(self, index):
            ...         return self.enable_mosaic
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.augment = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            return ret_val

        return wrapper


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
            cache_images=False, single_cls=False, stride=32, pad=0.0, prefix="", area_thr=0.2, ):
        super().__init__(augment=augment)
        self.img_size = img_size
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = (self.augment and not self.rect)  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None

        # additional feature
        self.area_thr = area_thr

        p = Path(path)  # os-agnostic
        self.img_files = self.get_img_files(p, prefix)
        self.label_files = img2label_paths(self.img_files)  # labels
        # Check cache
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        labels, shapes, segments, img_files, label_files = self.load_cache(cache_path, prefix)

        self.segments = segments
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = img_files  # update
        self.label_files = label_files  # update

        num_imgs = len(shapes)  # number of images
        batch_index = np.floor(np.arange(num_imgs) / batch_size).astype(np.int)  # batch index
        self.batch_index = batch_index  # batch index of image
        self.num_imgs = num_imgs
        self.indices = range(num_imgs)

        # Update labels
        for i, (_, segment) in enumerate(zip(self.labels, self.segments)):
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            num_batches = batch_index[-1] + 1  # number of batches
            self.update_rect(num_batches, pad)

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * num_imgs, [None] * num_imgs
        if cache_images:
            self.cache_images(cache_images, prefix)

    def cache_images(self, cache_images, prefix):
        """Cache images to disk or ram for faster speed."""
        if cache_images == "disk":
            self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + "_npy")
            self.img_npy = [self.im_cache_dir / Path(f).with_suffix(".npy").name for f in self.img_files]
            self.im_cache_dir.mkdir(parents=True, exist_ok=True)
        gb = 0  # Gigabytes of cached images
        self.img_hw0, self.img_hw = [None] * self.num_imgs, [None] * self.num_imgs
        results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(self.num_imgs)))
        pbar = tqdm(enumerate(results), total=self.num_imgs)
        for i, x in pbar:
            if cache_images == "disk":
                if not self.img_npy[i].exists():
                    np.save(self.img_npy[i].as_posix(), x[0])
                gb += self.img_npy[i].stat().st_size
            else:
                (self.imgs[i], self.img_hw0[i], self.img_hw[i],) = x  # im, hw_orig, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
            pbar.desc = f"{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})"
        pbar.close()

    def get_img_files(self, p, prefix):
        """Read image files."""
        try:
            f = []  # image files
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)  # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p, "r") as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in
                        t]  # local to global path  # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise Exception(f"{prefix}{p} does not exist")
            img_files = sorted([x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS])
            # img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert img_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {str(p)}: {e}\nSee {HELP_URL}")
        return img_files

    def load_cache(self, cache_path, prefix):
        """Load labels from *.cache file."""
        try:
            cache, exists = (np.load(cache_path, allow_pickle=True).item(), True,)  # load dict
            assert cache["version"] == self.cache_version  # same version
            assert cache["hash"] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache["msgs"]:
                logging.info("\n".join(cache["msgs"]))  # display warnings
        assert (
                nf > 0 or not self.augment), f"{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, segments = zip(*cache.values())
        img_files = list(cache.keys())  # update
        label_files = img2label_paths(cache.keys())  # update
        return labels, shapes, segments, img_files, label_files

    def update_rect(self, num_batches, pad):
        """Update attr if rect is True."""
        # Sort by aspect ratio
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.img_files = [self.img_files[i] for i in irect]
        self.label_files = [self.label_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.segments = [self.segments[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * num_batches
        for i in range(num_batches):
            ari = ar[self.batch_index == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = (np.ceil(np.array(shapes) * self.img_size / self.stride + pad).astype(np.int) * self.stride)

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Cache labels to *.cache file if there is no *.cache file in local."""
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = (0, 0, 0, 0, [],)  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."

        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix)), ),
                desc=desc, total=len(self.img_files), )
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info("\n".join(msgs))
        if nf == 0:
            logging.info(f"{prefix}WARNING: No labels found in {path}. See {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, len(self.img_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            logging.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            logging.info(f"{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}")  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        self.mosaic = self.augment and not self.rect
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.num_imgs - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = (
                self.batch_shapes[self.batch_index[index]] if self.rect else self.img_size)  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img, labels, degrees=hyp["degrees"], translate=hyp["translate"],
                    scale=hyp["scale"], shear=hyp["shear"], perspective=hyp["perspective"], )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts  # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, None

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = \
                F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False, )[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1),), 2, )
                l = (torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo,), 0, ) * s)
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
            cache_images=False, single_cls=False, stride=32, pad=0, prefix="", area_thr=0.2,
            downsample_ratio=1, overlap=False,
    ):
        super().__init__(path, img_size, batch_size, augment, hyp, rect, image_weights, cache_images, single_cls,
            stride, pad, prefix, area_thr, )
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        self.mosaic = self.augment and not self.rect
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        masks = []
        if mosaic:
            # Load mosaic
            img, labels, segments = load_mosaic(self, index, return_seg=True)
            shapes = None

            # TODO: Mixup not support segment for now
            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.num_imgs - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = (
                self.batch_shapes[self.batch_index[index]] if self.rect else self.img_size)  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()
            # TODO
            if len(segments):
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(segments[i_s], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1], )
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, segments = random_perspective(img, labels, segments=segments, degrees=hyp["degrees"],
                    translate=hyp["translate"], scale=hyp["scale"], shear=hyp["shear"], perspective=hyp["perspective"],
                    return_seg=True, )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                masks, sorted_idx = polygons2masks_overlap(img.shape[:2], segments, 
                        downsample_ratio=self.downsample_ratio)
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                labels = labels[sorted_idx]
            else:
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        masks = (torch.from_numpy(masks) if len(masks) else 
                torch.zeros(1 if self.overlap else nl, 
                    img.shape[0] // self.downsample_ratio,
                    img.shape[1] // self.downsample_ratio))
        # TODO: albumentations support
        if self.augment:
            # Albumentations
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
                    masks = torch.flip(masks, dims=[1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
                    masks = torch.flip(masks, dims=[2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return (torch.from_numpy(img), labels_out, self.img_files[index], shapes, masks)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, masks = zip(*batch)  # transposed
        batched_masks = torch.cat(masks, 0)
        # print(batched_masks.shape)
        # print('batched_masks:', (batched_masks > 0).sum())
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, "Image Not Found " + path
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR, )
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return (self.imgs[i], self.img_hw0[i], self.img_hw[i],)  # im, hw_original, hw_resized


def load_mosaic(self, index, return_seg=False):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y

    # 3 additional image indices
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        labels, segments = self.labels[index].copy(), self.segments[index].copy()

        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
    results = random_perspective(img4, labels4, segments4, degrees=self.hyp["degrees"], translate=self.hyp["translate"],
        scale=self.hyp["scale"], shear=self.hyp["shear"], perspective=self.hyp["perspective"],
        border=self.mosaic_border, area_thr=self.area_thr, return_seg=return_seg, )  # border to remove
    # return (img4, labels4, segments4) if return_seg else (img4, labels4)
    return results


def load_mosaic9(self, index):
    # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc: yc + 2 * s, xc: xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9, degrees=self.hyp["degrees"],
        translate=self.hyp["translate"], scale=self.hyp["scale"], shear=self.hyp["shear"],
        perspective=self.hyp["perspective"], border=self.mosaic_border, )  # border to remove

    return img9, labels9


def dataset_stats(path="coco128.yaml", autodownload=False, verbose=False, profile=False, hub=False):
    """Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith(".zip"):  # path is data.zip
            assert Path(path).is_file(), f"Error unzipping {path}, file not found"
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix("")  # dataset directory == zip name
            return (True, str(dir), next(dir.rglob("*.yaml")),)  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, quality=75)  # save
        except Exception as e:  # use OpenCV
            print(f"WARNING: HUB ops PIL failure {f}: {e}")
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_LINEAR, )
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors="ignore") as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data["path"] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data["path"] + ("-hub" if hub else ""))
    stats = {"nc": data["nc"], "names": data["names"]}  # statistics dictionary
    for split in "train", "val", "test":
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.num_imgs, desc="Statistics"):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data["nc"]))
        x = np.array(x)  # shape(128x80)
        stats[split] = {"instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
            "image_stats": {"total": dataset.num_imgs, "unlabelled": int(np.all(x == 0, 1).sum()),
                "per_class": (x > 0).sum(0).tolist(), },
            "labels": [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                zip(dataset.img_files, dataset.labels)], }

        if hub:
            im_dir = hub_dir / "images"
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.num_imgs,
                    desc="HUB Ops", ):
                pass

    # Profile
    stats_path = hub_dir / "stats.json"
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix(".npy")
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f"stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write")

            file = stats_path.with_suffix(".json")
            t1 = time.time()
            with open(file, "w") as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file, "r") as f:
                x = json.load(f)  # load hyps dict
            print(f"stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write")

    # Save, print and return
    if hub:
        print(f"Saving {stats_path.resolve()}...")
        with open(stats_path, "w") as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats


# REFACTOR IN NEW FILE
import os
import glob
import shutil
import hashlib
import torch
import cv2
import random
from pathlib import Path
from PIL import ImageOps, ExifTags
from utils.segment import segments2boxes
from utils.general import xywh2xyxy

# Parameters
HELP_URL = "https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data"
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo", ]  # acceptable image suffixes
VID_FORMATS = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv", "vdo", "flv", ]  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM, 5: Image.TRANSPOSE,
            6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90, }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image

def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly then resize is trying the keep the same way 
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], 
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color,
                            downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
                    dtype=np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = (os.sep + "images" + os.sep, os.sep + "labels" + os.sep,)  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def create_folder(path="./new"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path="../datasets/coco128"):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + "_flat")
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + "/**/*.*", recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path="../datasets/coco128", ):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / "classifier") if (path / "classifier").is_dir() else None  # remove existing
    files = list(path.rglob("*.*"))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, "r") as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32, )  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = ((path / "classifier") / f"{c}" / f"{path.stem}_{im_file.stem}_{j}.jpg")  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]: b[3], b[0]: b[2]]), f"box failure in {f}"


def autosplit(path="../datasets/coco128/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted([x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS])  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt", ]  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if (not annotated_only or Path(img2label_paths([str(img)])[0]).exists()):  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write("./" + img.relative_to(path.parent).as_posix() + "\n")  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = (0, 0, 0, 0, "", [],)  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING: {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, "r") as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 6 for x in l]):  # is segment
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert (l.shape[1] == 5), f"labels require 5 columns, {l.shape[1]} columns detected"
                assert (l >= 0).all(), f"negative label values {l[l < 0]}"
                assert (l[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}"
                l, idx = np.unique(l, axis=0, return_index=True)  # remove duplicate rows
                # NOTE: `np.unique` will change the order of `l`, so adjust the segments order too.
                segments = [segments[i] for i in idx] if len(segments) > 0 else segments
                if len(l) < nl:
                    msg = f"{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed"
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


from torch.utils.data import DataLoader as torchDataLoader


class DataLoader(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def close_augment(self):
        self.batch_sampler.augment = False


class InfiniteDataLoader(torchDataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
