# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Dataloaders."""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..augmentations import augment_hsv, copy_paste, letterbox
from ..dataloaders import InfiniteDataLoader, LoadImagesAndLabels, SmartDistributedSampler, seed_worker
from ..general import LOGGER, xyn2xy, xywhn2xyxy, xyxy2xywhn
from ..torch_utils import torch_distributed_zero_first
from .augmentations import mixup, random_perspective

RANK = int(os.getenv("RANK", -1))


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    mask_downsample_ratio=1,
    overlap_mask=False,
    seed=0,
):
    """
    Creates a dataloader for training, validating, or testing YOLO models with various dataset options.
    
    Args:
        path (str | list): Path to dataset directory or list of image files.
        imgsz (int): Size to which images should be resized.
        batch_size (int): Number of samples per batch.
        stride (int): Stride value for model input.
        single_cls (bool, optional): Whether the dataset consists of a single class. Defaults to False.
        hyp (dict, optional): Hyperparameters for augmentation. Defaults to None.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        cache (bool, optional): Whether to cache images for faster loading. Defaults to False.
        pad (float, optional): Padding value for images. Defaults to 0.0.
        rect (bool, optional): Whether to use rectangular training for faster training. Defaults to False.
        rank (int, optional): Rank of the process for distributed training. Defaults to -1.
        workers (int, optional): Number of subprocesses to use for data loading. Defaults to 8.
        image_weights (bool, optional): Whether to use weighted image sampling. Defaults to False.
        quad (bool, optional): Whether to use 4-mosaic augmentation. Defaults to False.
        prefix (str, optional): Prefix for logging. Defaults to "".
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        mask_downsample_ratio (int, optional): Downsample ratio for masks. Defaults to 1.
        overlap_mask (bool, optional): Whether to allow overlapping masks. Defaults to False.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    
    Returns:
        DataLoader or InfiniteDataLoader: Pytorch Dataloader object for iterating over the dataset.
    
    Notes:
        - Rectangular training is incompatible with DataLoader shuffling.
        - Adjusts the number of workers based on available CPU cores and batch size.
        - Uses different DataLoader classes depending on image_weights.
    
    Example:
        ```python
        dataloader = create_dataloader(
            path="path/to/dataset", imgsz=640, batch_size=16, stride=32, augment=True
        )
        for batch in dataloader:
            print(batch)
        ```
    """
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsAndMasks(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            downsample_ratio=mask_downsample_ratio,
            overlap=overlap_mask,
            rank=rank,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing
    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0,
        min_items=0,
        prefix="",
        downsample_ratio=1,
        overlap=False,
        rank=-1,
        seed=0,
    ):
        """
        Initializes the dataset with image, label, and mask loading capabilities for training/testing.
        
        Args:
            path (str): Path to the dataset directory.
            img_size (int): Size to which images should be resized. Defaults to 640.
            batch_size (int): Number of samples per batch. Defaults to 16.
            augment (bool): Whether to use dataset augmentation. Defaults to False.
            hyp (dict | None): Dictionary of hyperparameters for dataset augmentation. Defaults to None.
            rect (bool): If True, loads rectangular images. Defaults to False.
            image_weights (bool): Use weighted image selection for sampling. Defaults to False.
            cache_images (bool): Whether to cache images for faster loading. Defaults to False.
            single_cls (bool): Treat all labels as a single class. Defaults to False.
            stride (int): Image stride size. Defaults to 32.
            pad (float): Padding added to the image border. Defaults to 0.
            min_items (int): Minimum number of items in the dataset. Defaults to 0.
            prefix (str): Prefix for logging messages. Defaults to an empty string.
            downsample_ratio (float): Ratio for downsampling the masks. Defaults to 1.
            overlap (bool): Whether to use mask overlaps. Defaults to False.
            rank (int): Rank for distributed training. Defaults to -1.
            seed (int): Random seed for reproducibility. Defaults to 0.
        
        Returns:
            None
        
        Notes:
            For more information, refer to the official Ultralytics YOLOv5 GitHub repository: 
            https://github.com/ultralytics/yolov5
        
        Examples:
            ```python
            dataset = LoadImagesAndLabelsAndMasks(
                path='data/coco128.yaml',
                img_size=640,
                batch_size=16,
                augment=True,
                hyp={'hsv_h': 0.015},
                rect=False,
                image_weights=False,
                cache_images=True,
                single_cls=False,
                stride=32,
                pad=0.5,
                downsample_ratio=0.5,
                overlap=True,
                rank=0,
                seed=42,
            )
            ```
        """
        super().__init__(
            path,
            img_size,
            batch_size,
            augment,
            hyp,
            rect,
            image_weights,
            cache_images,
            single_cls,
            stride,
            pad,
            min_items,
            prefix,
            rank,
            seed,
        )
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap

    def __getitem__(self, index):
        """
        __getitem__(index: int) -> (tuple[np.ndarray, torch.Tensor, torch.Tensor, tuple]):
            """
            Returns a transformed item from the dataset at the specified index, including image, labels, and masks.
        
            Args:
                index (int): Index of the item to retrieve.
        
            Returns:
                tuple: A tuple containing four elements:
                    - img (np.ndarray): The processed image array in RGB format with shape (C, H, W).
                    - labels_out (torch.Tensor): Tensor containing label information of shape (num_labels, 6).
                    - masks (torch.Tensor): Tensor containing mask information of shape (1, H, W) or (num_labels, H, W).
                    - shapes (tuple): A tuple containing original image shapes and transformations applied.
        
            Notes:
                - Handles both mosaic and regular image loading, with multiple augmentation options.
                - Handles different shapes and formats for labels and masks.
                - Apply HSV augmentation, flipping, and perspective transformations if augmentations are enabled.
                - Refer to the original Ultralytics documentation for additional details: 
                  https://github.com/ultralytics/ultralytics
        
            Example:
                >>> dataset = LoadImagesAndLabelsAndMasks(...)
                >>> img, labels, masks, shapes = dataset[0]
        """
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        masks = []
        if mosaic:
            # Load mosaic
            img, labels, segments = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels, segments = mixup(img, labels, segments, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()
            if len(segments):
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, segments = random_perspective(
                    img,
                    labels,
                    segments=segments,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                masks, sorted_idx = polygons2masks_overlap(
                    img.shape[:2], segments, downsample_ratio=self.downsample_ratio
                )
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                labels = labels[sorted_idx]
            else:
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        masks = (
            torch.from_numpy(masks)
            if len(masks)
            else torch.zeros(
                1 if self.overlap else nl, img.shape[0] // self.downsample_ratio, img.shape[1] // self.downsample_ratio
            )
        )
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

        return (torch.from_numpy(img), labels_out, self.im_files[index], shapes, masks)

    def load_mosaic(self, index):
        """
        Loads a mosaic image composed of 1 image + 3 randomly chosen images, adjusting labels and segments.
        
        Args:
            index (int): The index of the primary image to be loaded.
        
        Returns:
            tuple[np.ndarray, np.ndarray, list]: A tuple containing the mosaic image, its labels, and segments.
            
        Notes:
            This method is used to create mosaic inputs for training, where four images are combined into one,
            and their labels and segments are appropriately resized and positioned. This is useful for data augmentation
            strategies to improve model robustness and generalization.
            
        Example:
            ```python
            index = 0
            img, labels, segments = obj.load_mosaic(index)
            # img: Numpy array of the combined mosaic image
            # labels: Numpy array of combined labels adjusted for the mosaic image
            # segments: List of combined segments adjusted for the mosaic image
            ```
        """
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

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
        img4, labels4, segments4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove
        return img4, labels4, segments4

    @staticmethod
    def collate_fn(batch):
        """
        Custom collation function for DataLoader, batches images, labels, paths, shapes, and segmentation masks.
        
        Args:
            batch (list[tuple[torch.Tensor, torch.Tensor, str, tuple, torch.Tensor]]): A list of tuples, where each tuple 
                contains an image tensor, a label tensor, a path string, a shape tuple, and a segmentation mask tensor.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]: A tuple containing:
                - Batched images (torch.Tensor)
                - Batched labels with index (torch.Tensor)
                - Image paths (list[str])
                - Shapes for scaling (torch.Tensor)
                - Batched segmentation masks (torch.Tensor)
        
        Note:
            The function ensures all elements in the batch are appropriately stacked and dimensioned for model input.
        
        Example:
            ```python
            from ultralytics import LoadImagesAndLabelsAndMasks
            from torch.utils.data import DataLoader
        
            dataset = LoadImagesAndLabelsAndMasks('path/to/dataset', img_size=640, batch_size=16)
            dataloader = DataLoader(dataset, batch_size=16, collate_fn=LoadImagesAndLabelsAndMasks.collate_fn)
            ```
        """
        img, label, path, shapes, masks = zip(*batch)  # transposed
        batched_masks = torch.cat(masks, 0)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks


def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Converts a list of polygons to a binary mask.
    
    Args:
        img_size (tuple[int, int]): Size of the image as a tuple (height, width).
        polygons (np.ndarray): Array of shape (N, M), where N is the number of polygons, and M (divisible by 2) represents
            the number of coordinates (x, y) for each polygon.
        color (int or float, optional): Fill color for the binary mask. Defaults to 1.
        downsample_ratio (int, optional): Ratio by which to downsample the mask. Defaults to 1.
    
    Returns:
        np.ndarray: A binary mask of the same height and width as the original image, optionally downsampled.
    
    Notes:
        The function creates a blank mask of the specified image size, fills in the specified polygons,
        and then downsamples the mask if a downsample_ratio is provided.
    
    Examples:
        ```python
        img_size = (1280, 720)
        polygons = np.array([[50, 50, 150, 50, 150, 150, 50, 150], [200, 200, 300, 200, 300, 300, 200, 300]])
        mask = polygon2mask(img_size, polygons)
        ```
    
    Returns:
        np.ndarray: A binary mask where the specified polygons are filled with the given color and the rest of the area is 0.
    ```
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Converts a list of polygons to masks suitable for segmentation tasks.
    
    Args:
        img_size (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons where each polygon is represented as an (N, M) array.
            N denotes the number of polygons, and M represents the number of points (should be divisible by 2).
        color (int | tuple): The color used to fill the polygons in the mask. Can be an integer or a tuple of RGB values.
        downsample_ratio (int, optional): The ratio to downsample the generated masks. Defaults to 1 (no downsampling).
    
    Returns:
        np.ndarray: An array of shape (len(polygons), img_size[0] // downsample_ratio, img_size[1] // downsample_ratio), where each slice along the first dimension represents a downsampled mask for the corresponding polygon.
    
    Examples:
        ```python
        img_size = (640, 640)
        polygons = [np.array([[100, 120, 140, 160, 180, 200]]), np.array([[200, 220, 240, 260, 280, 300]])]
        masks = polygons2masks(img_size, polygons, color=1, downsample_ratio=2)
        ```
        This example creates masks for two polygons and downsamples them by a factor of 2.
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """
    Creates an overlap mask from segmented polygons.
    
    Args:
        img_size (tuple): The dimensions of the image as (height, width).
        segments (list[np.ndarray]): A list of arrays where each array represents a polygon with an Nx2 format, denoting the 
            coordinates of the polygon's vertices.
        downsample_ratio (int, optional): Factor by which to downsample the output mask. Default is 1.
    
    Returns:
        np.ndarray: A 2D array representing the overlap mask, with dimensions proportional to `img_size` based on 
            `downsample_ratio`.
        np.ndarray: An array of indices that indicates the sorted order of the polygons based on their areas, in descending order.
    
    Notes:
        This function generates an overlap mask where overlapping regions are labeled with the index of the overlapping polygon. 
        Regions with multiple overlaps will be labeled with the index of the most recent polygon in the sorted order. This is particularly 
        useful for distinguishing overlapping segments during image processing tasks.
    
    Examples:
        ```python
        img_size = (640, 640)
        segments = [np.array([(10, 10), (100, 10), (100, 100), (10, 100)]), np.array([(50, 50), (150, 50), (150, 150), (50, 150)])]
        mask, sorted_indices = polygons2masks_overlap(img_size, segments, downsample_ratio=2)
        ```
    """
    masks = np.zeros(
        (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
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
