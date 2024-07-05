# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Dataloaders and dataset utils."""

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    check_dataset,
    check_requirements,
    check_yaml,
    clean_str,
    cv2,
    is_colab,
    is_kaggle,
    segments2boxes,
    unzip_file,
    xyn2xy,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = "See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(paths):
    """
    Generates a single SHA256 hash for a list of file or directory paths by combining their sizes and paths.
    
    Args:
        paths (list[str]): List of file or directory paths to be hashed.
    
    Returns:
        str: SHA256 hash of the file or directory paths.
    
    Note:
        The function sums the sizes of the existing paths and combines this with the concatenated string of all paths to generate a unique hash value. This can be useful for cache invalidation or verifying the consistency of a set of files.
    
    Uses:
        - `os.path.getsize`: To get the size of each path.
        - `hashlib.sha256`: To compute the SHA256 hash.
    
    Example:
        ```python
        paths = ["path/to/file1", "path/to/file2", "path/to/directory"]
        hash_value = get_hash(paths)
        print(f"SHA256 Hash: {hash_value}")
        ```
    """
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """
    Returns corrected image size considering EXIF orientation.
    
    Args:
        img (PIL.Image.Image): The input image to get the size of.
    
    Returns:
        tuple[int, int]: A tuple containing the corrected image size as (width, height).
    
    Notes:
        The function retrieves the EXIF orientation tag to determine if the image should be rotated for correct 
        size calculation. If the image lacks EXIF data or fails to provide orientation information, the original 
        size is returned.
    
    Examples:
        ```python
        from PIL import Image
        from utils.datasets import exif_size
    
        img = Image.open('image_with_exif.jpg')
        width, height = exif_size(img)
        ```
    
    Refer to: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def exif_transpose(image):
    """
    Transpose a PIL image based on its EXIF Orientation tag.
    
    Args:
        image (PIL.Image.Image): The image to transpose.
    
    Returns:
        PIL.Image.Image: The transposed image.
    
    Note:
        This function modifies the image in place if it has an EXIF Orientation tag.
        For more details, refer to: https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py#L375
    
    Example:
        ```python
        from PIL import Image
        from utils import exif_transpose
    
        img = Image.open('image.jpg')
        transposed_image = exif_transpose(img)
        transposed_image.show()
        ```
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def seed_worker(worker_id):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.
    
    Args:
        worker_id (int): The ID of the worker process. Although this parameter is not used in the function, it is required
        by PyTorch's API for worker initialization functions.
    
    Returns:
        None
    
    See Also:
        For more details on PyTorch's handling of randomness and the benefits of setting seeds, refer to 
        https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    
    Example:
        To ensure reproducibility when creating DataLoader instances in PyTorch, use the seed_worker function in conjunction
        with a DataLoader's `worker_init_fn` parameter:
    
        ```python
        from torch.utils.data import DataLoader
    
        loader = DataLoader(dataset, worker_init_fn=seed_worker)
        ```
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Inherit from DistributedSampler and override iterator
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
class SmartDistributedSampler(distributed.DistributedSampler):
    def __iter__(self):
        """
        Yields indices for distributed data sampling, shuffled deterministically based on epoch and seed.
        
        Returns:
            Iterator[int]: An iterator over shuffled sample indices for distributed training.
        
        Notes:
            This function is an override of the `__iter__` method in `torch.utils.data.distributed.DistributedSampler`.
            It establishes a deterministic approach to index shuffling based on a combination of seed and epoch, making it
            suitable for consistent results during distributed training. Shuffle behavior and dropping of last samples are
            dependent on the class attributes `shuffle` and `drop_last`, respectively.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # determine the eventual size (n) of self.indices (DDP indices)
        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1  # num_replicas == WORLD_SIZE
        idx = torch.randperm(n, generator=g)
        if not self.shuffle:
            idx = idx.sort()[0]

        idx = idx.tolist()
        if self.drop_last:
            idx = idx[: self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx)))[:padding_size]

        return iter(idx)


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
    seed=0,
):
    """
    Creates and returns a configured DataLoader instance for loading and processing image datasets.
    
    Args:
        path (str | Path): Path to dataset directory.
        imgsz (int): Image size for dataset images.
        batch_size (int): Number of samples per batch to load.
        stride (int): Model stride size.
        single_cls (bool): Treat the dataset as a single class. Defaults to False.
        hyp (dict): Hyperparameters for dataset augmentation. Defaults to None.
        augment (bool): Apply dataset augmentations. Defaults to False.
        cache (bool): Cache images to memory for faster training. Defaults to False.
        pad (float): Padding value for image padding. Defaults to 0.0.
        rect (bool): Load images in rectangular shapes for each batch. Defaults to False.
        rank (int): Process rank for distributed training. Defaults to -1.
        workers (int): Number of CPU workers to use for data loading. Defaults to 8.
        image_weights (bool): Weight images by their labels. Defaults to False.
        quad (bool): Load images four at a time. Defaults to False.
        prefix (str): Prefix for logging messages. Defaults to an empty string.
        shuffle (bool): Shuffle the dataset. Defaults to False.
        seed (int): Random seed to ensure reproducibility. Defaults to 0.
    
    Returns:
        DataLoader: torch DataLoader instance configured for the specified dataset.
    
    Notes:
        - See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data for more details.
        - If `rect` is True and `shuffle` is True, `shuffle` will be set to False with a warning message.
        - Uses a smart distributed sampler if `rank` is not -1 to ensure distributed training capabilities.
    
    Example:
        ```python
        from ultralytics.utils.dataloaders import create_dataloader
    
        dataloader = create_dataloader(
            path='data/coco128.yaml',
            imgsz=640,
            batch_size=16,
            stride=32,
            single_cls=False,
            hyp=None,
            augment=True,
            cache=False,
            pad=0.0,
            rect=False,
            rank=-1,
            workers=8,
            image_weights=False,
            quad=False,
            prefix='',
            shuffle=True,
            seed=42,
        )
        ```
    """
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
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
        pin_memory=PIN_MEMORY,
        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a repeating sampler.
        
        Args:
            *args (any): Variable length argument list that is passed to the superclass DataLoader.
            **kwargs (any): Arbitrary keyword arguments that are passed to the superclass DataLoader.
        
        Returns:
            None
        
        Notes:
            The `InfiniteDataLoader` modifies the standard PyTorch DataLoader to reuse workers and implement an infinite
            looping functionality via a custom sampler (`_RepeatSampler`), ensuring a continuous stream of data.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """
        Returns the length of the batch sampler's sampler.
        
        Returns:
            int: Length of the batch sampler's sampler.
        """
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """
        Iterates over batches of data indefinitely by resetting the sampler when it is exhausted.
        
        Returns:
            Iterator: An iterator that yields batches of data endlessly.
        
        Examples:
            ```python
            # Example usage
            dataloader = InfiniteDataLoader(dataset, batch_size=32, shuffle=True)
            for batch in dataloader:
                # process batch
            ```
        """
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """
        Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration.
        
        Args:
            sampler (torch.utils.data.Sampler): A PyTorch Sampler instance that defines the sampling strategy for the data.
        
        Returns:
            None: This method does not return any value.
        
        Notes:
            This class is used internally by `InfiniteDataLoader` to provide continuous data iteration without the need
            to reinitialize the data loader.
        
        Example:
            ```python
            from torch.utils.data import DataLoader, RandomSampler
            from ultralytics.dataloaders import _RepeatSampler, InfiniteDataLoader
        
            dataset = ...  # your dataset
            sampler = RandomSampler(dataset)
            repeat_sampler = _RepeatSampler(sampler)
            infinite_loader = InfiniteDataLoader(dataset, batch_size=32, sampler=repeat_sampler)
        
            for data in infinite_loader:
                # process your data
            ```
        """
        self.sampler = sampler

    def __iter__(self):
        """
        Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler.
        
        Returns:
            Iterator: An iterator that infinitely yields samples from the provided sampler's data.
        
        Notes:
            The method is designed to facilitate continuous training iterations without manually resetting the data loader.
        
        Example:
            ```python
            from torch.utils.data import DataLoader, RandomSampler
            from ultralytics import _RepeatSampler
        
            dataset = MyDataset()
            sampler = RandomSampler(dataset)
            repeat_sampler = _RepeatSampler(sampler)
        
            for data in repeat_sampler:
                # process your data here
            ```
        """
        while True:
            yield from iter(self.sampler)


class LoadScreenshots:
    # YOLOv5 screenshot dataloader, i.e. `python detect.py --source "screen 0 100 100 512 256"`
    def __init__(self, source, img_size=640, stride=32, auto=True, transforms=None):
        """
        Initializes a screenshot dataloader for YOLOv5 with a specified source region, image size, stride, auto, and transforms.
        
        Args:
            source (str): Region coordinates for capturing the screen, formatted as "screen_number left top width height".
            img_size (int, optional): Target image size after preprocessing. Defaults to 640.
            stride (int, optional): Stride size for the model. Defaults to 32.
            auto (bool, optional): Flag for automatic adjustment of image size and stride. Defaults to True.
            transforms (callable | None, optional): Optional image transformation function. Defaults to None.
        
        Returns:
            None
        
        Example:
            ```python
            dataloader = LoadScreenshots("0 100 100 512 256", img_size=640, stride=32, auto=True, transforms=None)
            ```
        """
        check_requirements("mss")
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = "stream"
        self.frame = 0
        self.sct = mss.mss()

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        """
        Iterates over the screenshots, capturing and processing each frame according to initialization parameters.
        
        Yields:
            tuple[np.ndarray, str, False]: A tuple containing the captured frame in NumPy array format, an empty string as a
            placeholder for filename, and a boolean `False` indicating not the end of the stream.
        
        Note:
            Required package: mss (https://python-mss.readthedocs.io/).
        
        Example:
            ```python
            loader = LoadScreenshots("screen 0 100 100 512 256", img_size=640, stride=32)
            for img, _, _ in loader:
                # Process the img
            ```
        """
        return self

    def __next__(self):
        """
        __next__()
        Captures and processes the next screenshot frame, returning it as a numpy array ready for model input.
        
        Returns:
            np.ndarray: The processed frame as a numpy array in CHW format with BGR channels.
        
        Raises:
            StopIteration: Raised when the screenshot capture process is manually stopped.
        
        Example:
            ```python
            screenshots_loader = LoadScreenshots(source="screen 0 100 100 512 256", img_size=640)
            for frame in screenshots_loader:
                # Process the frame
                pass
            ```
        """
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s


class LoadImages:
    """YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`"""

    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """
        Initializes the YOLOv5 `LoadImages` class, supporting image and video loading from various sources including glob patterns, directories, and files.
        
        Args:
            path (str | list[str]): Path to the image or video file, directory, or glob pattern. If a .txt file is provided, it should contain a list of image/video paths.
            img_size (int, optional): Target image size for resizing. Defaults to 640.
            stride (int, optional): Stride value for resizing. Defaults to 32.
            auto (bool, optional): Whether to use automatic aspect ratio adjustment in resizing. Defaults to True.
            transforms (callable, optional): Function to apply custom transformations to the loaded images. Defaults to None.
            vid_stride (int, optional): Frame-rate stride value for video loading. Defaults to 1.
        
        Raises:
            FileNotFoundError: If the provided path does not exist or is not a valid file, directory, or glob pattern.
        
        Returns:
            None
        
        Examples:
            ```python
            loader = LoadImages('data/images', img_size=640, stride=32)
            for path, img, img0s, vid_cap, s in loader:
                # process img
            ```
        
        Notes:
            - Supported image formats include: bmp, dng, jpeg, jpg, mpo, png, tif, tiff, webp, pfm
            - Supported video formats include: asf, avi, gif, m4v, mkv, mov, mp4, mpeg, mpg, ts, wmv
            - Useful for loading images and videos for inference in YOLOv5. For further details, refer to
              the [YOLOv5 documentation](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data).
        """
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        """
        __iter__()
        Returns an iterator that yields image or video frames with associated metadata.
        
        Yields:
            tuple[str, ndarray, ndarray | None, VideoCapture | None, str]: A tuple containing:
                - The file path or video frame identifier (str)
                - The transformed image or frame (ndarray)
                - The original (letterboxed) image or frame (ndarray | None)
                - The video capture object if applicable, else None (VideoCapture | None)
                - The filename with dimensions (str)
        
        Example:
            ```python
            dataloader = LoadImages(path="path/to/data")
            for file_path, img, img0, cap, s in dataloader:
                print(file_path, img.shape, cap, s)
            ```
        """
        self.count = 0
        return self

    def __next__(self):
        """
        __next__()
            Returns the next image or video frame in the dataset.
        
            When iterating over a collection of images or video frames, this method will handle reading the next item
            in the sequence, applying necessary transformations, and returning the processed result along with additional
            metadata.
        
            Returns:
                (tuple[str, np.ndarray, np.ndarray, None, str]): A tuple containing:
                    - Path or description of the current media item, as a string.
                    - Processed image/frame as a NumPy array.
                    - Original image/frame as a NumPy array.
                    - Placeholder for future use (currently None).
                    - Status message as a string.
                    
            Raises:
                StopIteration: When the end of the dataset is reached.
        """
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """
        Initializes a new video capture object with path, frame count adjusted by stride, and orientation metadata.
        
        Args:
            path (str): Path to the video file to be opened.
        
        Returns:
            None
        
        Note:
            This method sets up the video capture and configuration for processing video frames, 
            adjusting the frame count based on the specified video stride.
        
        Examples:
            ```python
            loader = LoadImages("video.mp4")
            loader._new_video("new_video.mp4")
            ```
        """
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        """
        Rotates a cv2 image based on its orientation metadata; supports 0, 90, and 180 degrees rotations.
        
        Args:
            im (np.ndarray): The input image in ndarray format.
            orientation (int): The orientation metadata of the image (0, 90, 180, 270 degrees).
        
        Returns:
            np.ndarray: The rotated image according to the orientation.
        
        Examples:
            ```python
            rotated_image = self._cv2_rotate(input_image)
            ```
        
        Notes:
            Ensure the orientation metadata is correctly provided; otherwise, the image may not rotate as expected.
        """
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """
        Returns the total number of image and video files in the dataset.
        
        Returns:
            int: The number of image and video files in the dataset.
        """
        return self.nf  # number of files


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources="file.streams", img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """
        Initializes a stream loader for processing video streams with YOLOv5, supporting various sources including YouTube.
        
        Args:
            sources (str): Path to a file containing stream URLs or a single URL string.
            img_size (int): Desired image size after resizing. Default is 640.
            stride (int): Stride value for letterbox transformations. Default is 32.
            auto (bool): If True, uses automatic stride calculation for resizing. Default is True.
            transforms (callable | None): Optional transform function to apply to each frame.
            vid_stride (int): Specifies the frame-rate stride for video processing. Default is 1.
        
        Returns:
            None
        
        Notes:
            This function initializes necessary configurations and starts threads for reading video streams. It supports various
            video sources including local webcams, RTSP, RTMP, HTTP streams, and YouTube videos.
        
        Example:
            ```python
            loader = LoadStreams(sources="https://www.youtube.com/watch?v=example", img_size=640, stride=32)
            for img in loader:
                # process each image frame
            ```
        
        For more information on stream sources, see https://docs.ultralytics.com/yolov5/tutorials/train_custom_data.
        """
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "
            if urlparse(s).hostname in ("www.youtube.com", "youtube.com", "youtu.be"):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
                check_requirements(("pafy", "youtube_dl==2020.12.2"))
                import pafy

                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), "--source 0 webcam unsupported on Colab. Rerun command in a local environment."
                assert not is_kaggle(), "--source 0 webcam unsupported on Kaggle. Rerun command in a local environment."
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"{st}Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")  # newline

        # check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning("WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.")

    def update(self, i, cap, stream):
        """
        Updates frames from each stream in a multithreaded environment, ensuring continuous frame retrieval.
        
        Args:
            i (int): Index of the stream in the list of sources.
            cap (cv2.VideoCapture): OpenCV VideoCapture object for the given stream.
            stream (str): The source stream URL or local path.
        
        Returns:
            None
        
        Notes:
            - This method should be used as a target for multithreading to keep reading frames from each video stream
              concurrently.
            - If a stream becomes unresponsive, it attempts to reopen the stream to restore frame capture continuity.
        """
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        """
        __iter__()
            """Resets and returns the iterator for iterating over video frames or images in a dataset.
        
            Yields:
                (tuple[str, np.ndarray, np.ndarray, None, str]): Information about the current frame, including:
                    - Source identifier as a string.
                    - Transformed image as a numpy array in CHW format, RGB order.
                    - Original image as a numpy array in HWC format, BGR order.
                    - Stream capture object (always None).
                    - Status string with detailed descriptions.
            
            Examples:
                ```python
                stream_loader = LoadStreams(sources="path/to/stream")
                for source, img, img0, cap, status in stream_loader:
                    # Process image frame here
                ```
            """
            self.count = -1
            return self
        """
        self.count = -1
        return self

    def __next__(self):
        """
        Returns the next processed frame from the stream.
        
        Returns:
            (tuple[str, ndarray, ndarray, None, str]): A tuple containing the following elements:
                - Source name or index of the video stream (str).
                - Processed image as a numpy array in contiguous RGB format (ndarray).
                - Original image in BGR format captured from the stream (ndarray).
                - Always None for compatibility (None).
                - Status string with stream details (str).
        """
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ""

    def __len__(self):
        """
        Returns the number of video streams being processed.
        
        Returns:
            int: The number of video streams in the `LoadStreams` instance.
        """
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    """
    Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and extension
    with `.txt`.
    
    Args:
      img_paths (list[str]): List of image file paths for which to generate corresponding label file paths.
    
    Returns:
      list[str]: List of label file paths generated from image paths by modifying directory and extension.
    
    Examples:
      ```python
      img_paths = ['/path/to/images/img1.jpg', '/path/to/images/img2.png']
      label_paths = img2label_paths(img_paths)
      # label_paths = ['/path/to/labels/img1.txt', '/path/to/labels/img2.txt']
      ```
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

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
        pad=0.0,
        min_items=0,
        prefix="",
        rank=-1,
        seed=0,
    ):
        """
        Initializes the YOLOv5 dataset loader with image paths and label data, supporting augmentation, caching, and preprocessing.
        
        Args:
            path (str | list): Path to the dataset directory, a text file with paths, or a list of paths.
            img_size (int): Desired image size after preprocessing, typically used for model training.
            batch_size (int): The number of samples per batch of input images.
            augment (bool): If True, applies image augmentations for training.
            hyp (dict, optional): Hyperparameters for data augmentation if augment is True.
            rect (bool): If True, uses rectangular training which maintains the aspect ratio of images within the batch.
            image_weights (bool): If True, uses image weights to balance the dataset classes.
            cache_images (bool | str): If True or "ram", caches images in RAM; if "disk", caches images on disk.
            single_cls (bool): If True, considers all classes as a single class (class 0).
            stride (int): The stride for resizing images during data preprocessing.
            pad (float): Padding added to each image.
            min_items (int): Minimum items (images/labels) in a category to be included in the dataset.
            prefix (str): A prefix for logging and error messages.
            rank (int): Rank for distributed training; used for Data Parallelism (DP).
            seed (int): Random seed for deterministic behavior in distributed training environments.
        
        Returns:
            None
        
        Notes:
            This method also handles the loading of labels, image caching, and initializing data structures required for the
            YOLOv5 training pipeline. Use the `see: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data` for detailed
            training data preparation guidelines.
        
        Examples:
            ```python
            dataloader = LoadImagesAndLabels(path='data/coco128.yaml', img_size=640, batch_size=16, augment=True, hyp=hyp,
                                             rect=False, image_weights=False, cache_images=False, single_cls=False, stride=32,
                                             pad=0.0, min_items=0, prefix='', rank=0, seed=42)
            ```
        """
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}\n{HELP_URL}") from e

        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f"{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f"{prefix}{n - len(include)}/{n} images filtered from dataset")
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)
        if rank > -1:  # DDP indices (see: SmartDistributedSampler)
            # force each rank (i.e. GPU process) to sample the same subset of data on every epoch
            self.indices = self.indices[np.random.RandomState(seed=seed).permutation(n) % WORLD_SIZE == RANK]

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            results = ThreadPool(NUM_THREADS).imap(lambda i: (i, fcn(i)), self.indices)
            pbar = tqdm(results, total=len(self.indices), bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes * WORLD_SIZE
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"
            pbar.close()

    def check_cache_ram(self, safety_margin=0.1, prefix=""):
        """
        Checks if available RAM is sufficient for caching images, considering a safety margin.
        
        Args:
            safety_margin (float, optional): Proportion of available RAM to reserve as a safety margin. Defaults to 0.1.
            prefix (str, optional): String to prefix log messages with. Defaults to "".
        
        Returns:
            bool: True if there is sufficient RAM to cache the images, False otherwise.
        
        Notes:
            Caching images can significantly speed up training by reducing disk I/O but requires sufficient available RAM.
            
        Examples:
            ```python
            dataset = LoadImagesAndLabels(path='data/images', img_size=640, augment=True)
            can_cache = dataset.check_cache_ram(safety_margin=0.2, prefix='TRAIN:')
            if can_cache:
                # Proceed with caching images to RAM
            else:
                # Handle limited RAM availability
            ```
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """
        Caches dataset labels, verifies images, reads shapes, and tracks dataset integrity.
        
        Args:
            path (Path, optional): Path to save the cache file. Defaults to Path("./labels.cache").
            prefix (str, optional): Prefix for log messages. Defaults to "".
        
        Returns:
            dict: Dictionary containing the cached dataset labels, file hash, results, messages, and cache version.
        
        Notes:
            - Caches dataset labels to speed up subsequent training sessions, reducing redundant I/O operations.
            - Verifies image files and dataset integrity to ensure labels and images are correctly paired and readable.
        
        Example usage:
        ```python
        loader = LoadImagesAndLabels(path)
        cache = loader.cache_labels()
        ```
        """
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                desc=desc,
                total=len(self.im_files),
                bar_format=TQDM_BAR_FORMAT,
            )
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def __len__(self):
        """
        Returns the number of images in the dataset.
        
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.im_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        """
        Fetches a dataset item by index, performing necessary data augmentations and preprocessing.
        
        Args:
            index (int): The index of the dataset item to fetch.
        
        Returns:
            tuple: A tuple containing:
                - img (numpy.ndarray): The preprocessed image array of shape `(3, H, W)`, where H and W are image dimensions.
                - labels_out (torch.Tensor): The tensor of shape `(N, 6)` containing ground truth boxes and other information for N objects.
                - self.im_files[index] (str): File path of the image.
                - shapes (None | tuple): If `rect` parameter is True, returns a tuple containing original and resized shape details.
        
        Notes:
            - Supports mosaic and mixup augmentations.
            - Applies letterboxing, random perspective transformations, HSV augmentations, and flipping augmentations.
            - Transforms bounding boxes to the desired format, handling different augmentation scenarios.
            - Caches precomputed images and labels, enhancing data loading efficiency in subsequent epochs.
        
        Example:
            ```python
            dataset = LoadImagesAndLabels(path, img_size=640, batch_size=16, augment=True)
            img, labels_out, img_path, shapes = dataset[0]
            ```
        """
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *self.load_mosaic(random.choice(self.indices)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

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

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        """
        Loads an image by its index, returning the image array, its original dimensions, and resized dimensions.
        
        Args:
            i (int): Index of the image to load.
        
        Returns:
            tuple[np.ndarray, tuple[int, int], tuple[int, int]]: A tuple containing the loaded image, its original dimensions
            (height, width), and its resized dimensions (height, width).
        
        Raises:
            AssertionError: If the image file is not found.
        
        Notes:
            - Cached images in RAM or on disk are used if available.
            - Resizing maintains the aspect ratio, interpolating as needed.
        
        Example:
            ```python
            img, original_hw, resized_hw = dataset.load_image(0)
            ```
        """
        im, f, fn = (
            self.ims[i],
            self.im_files[i],
            self.npy_files[i],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        """
        Caches images to disk as *.npy files for faster subsequent loading.
        
        Args:
            i (int): The index of the image to be cached to disk.
        
        Returns:
            None
        
        Notes:
            This function will try to save the image at the given index `i` as a `.npy` file on the disk for quicker access in the future.
            It does so only if the file does not already exist on disk. The function reads the image, resizes it if necessary,
            and saves it using numpy's `save` function. This allows for quicker loading during next access.
        
        Examples:
            ```python
            dataset = LoadImagesAndLabels(path='data/images', cache_images='disk')
            dataset.cache_images_to_disk(0)
            ```
        
            This example initializes a dataset and caches the first image to disk.
        """
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        """
        Loads a 4-image mosaic for YOLOv5, combining 1 selected and 3 random images, with labels and segments.
        
        Args:
            index (int): Index of the main image to be used in the mosaic.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the mosaic image and the combined labels for the mosaic.
            
        Notes:
            This method combines four images into one. The selected image is placed in one of the four quadrants along with 
            three other randomly selected images, effectively increasing the variety and complexity of the training data, and 
            enhancing data augmentation. The image is transformed through various augmentations including scaling, rotating, 
            and translating, based on specified hyperparameters. Labels are also transformed accordingly.
        """
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
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

            # Labels
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
        img4, labels4 = random_perspective(
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

        return img4, labels4

    def load_mosaic9(self, index):
        """
        LoadImagesAndLabels.load_mosaic9(index)
        """Loads an image mosaic composed of 9 different images for augmented YOLOv5 training.
        
        This function combines 1 selected image with 8 randomly chosen supplementary images from the dataset, forming a 
        3x3 mosaic grid. Each image's labels and segments are accurately transformed and clipped to match the augmented 
        mosaic.
        
        Args:
            index (int): The index of the selected base image around which the mosaic is created.
        
        Returns:
            tuple:
                - img9 (np.ndarray): Mosaic image of shape (3 * img_size, 3 * img_size, channels).
                - labels9 (np.ndarray): Labels corresponding to the combined mosaic image.
                - segments9 (List[np.ndarray]): Segmented regions of objects within the mosaic image.
        """
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

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
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady :, x1 - padx :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc : yc + 2 * s, xc : xc + 2 * s]

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
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp["copy_paste"])
        img9, labels9 = random_perspective(
            img9,
            labels9,
            segments9,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        """
        Loads and merges a batch of images and labels into a single tensor.
        
        Args:
            batch (list of tuple): A list where each tuple contains an image tensor, a label tensor, the image path, and its
                shape. The label tensor is expected to have the shape `(num_labels, 6)` where the initial dimension represents the 
                image index and bounding box parameters.
        
        Returns:
            tuple:
                - torch.Tensor: Batch of images stacked into a tensor.
                - torch.Tensor: Merged labels tensor, where each label includes an image index.
                - list of str: List of image paths.
                - list of tuple: Shapes of the original images.
        """
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        """
        Collate a batch's data by quartering the number of shapes and paths, preparing it for model input.
        
        Args:
          batch (list[tuple[torch.Tensor, torch.Tensor, str, tuple]]): 
              A list of tuples where each tuple contains:
              - im (torch.Tensor): The image tensor.
              - label (torch.Tensor): The label tensor.
              - path (str): The image path.
              - shapes (tuple): The shapes of images.
        
        Returns:
          tuple[torch.Tensor, torch.Tensor, tuple, tuple]: 
              A tuple containing:
              - im4 (torch.Tensor): The combined image tensor.
              - label4 (torch.Tensor): The combined label tensor.
              - path4 (tuple): The first quarter of the paths.
              - shapes4 (tuple): The first quarter of the shapes.
        """
        im, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False)[
                    0
                ].type(im[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im1)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def flatten_recursive(path=DATASETS_DIR / "coco128"):
    """
    Flattens a directory by copying all files from subdirectories to a new top-level directory, preserving filenames.
    
    Args:
        path (str | Path): Root directory path to flatten. Defaults to 'coco128' dataset path.
    
    Returns:
        None: This function performs file operations and does not return a value.
    
    Note:
        If the new flattened directory already exists, it will be deleted and recreated to prevent conflicts or duplication.
    
    Example:
        ```python
        flatten_recursive('/path/to/dataset')
        ```
        This will create a new directory '/path/to/dataset_flat' with all files from the nested structure inside '/path/to/dataset' copied to it.
    """
    new_path = Path(f"{str(path)}_flat")
    if os.path.exists(new_path):
        shutil.rmtree(new_path)  # delete output folder
    os.makedirs(new_path)  # make new output folder
    for file in tqdm(glob.glob(f"{str(Path(path))}/**/*.*", recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / "coco128"):
    """
    Converts a detection dataset to a classification dataset by extracting bounding boxes and saving them as individual images.
    
    Args:
        path (str | pathlib.Path): Path to the directory containing the dataset in detection format.
    
    Returns:
        None
    
    Example:
        ```python
        from utils.dataloaders import extract_boxes
        extract_boxes('path/to/dataset')
        ```
    
    Note:
        This function will create a new directory named "classification" within the provided path. Each class will have its own subdirectory, containing images of the respective bounding boxes. Existing "classification" directory, if any, will be removed prior to extraction.
    """
    path = Path(path)  # images dir
    shutil.rmtree(path / "classification") if (path / "classification").is_dir() else None  # remove existing
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
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / "classification") / f"{c}" / f"{path.stem}_{im_file.stem}_{j}.jpg"  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1] : b[3], b[0] : b[2]]), f"box failure in {f}"


def autosplit(path=DATASETS_DIR / "coco128/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
    Autosplit a dataset into train, validation, and test splits by creating corresponding text files with image paths.
    
    Args:
        path (str | Path, optional): Path to the image directory to be split. The default is the `coco128/images` directory.
        weights (tuple[float, float, float], optional): Proportions for the train, validation, and test splits. The default is (0.9, 0.1, 0.0).
        annotated_only (bool, optional): If True, include only images with accompanying annotation files. Defaults to False.
    
    Returns:
        None
    
    Notes:
        The function outputs three text files named `autosplit_train.txt`, `autosplit_val.txt`, and `autosplit_test.txt` containing the respective image paths.
    
    Examples:
        ```python
        from utils.dataloaders import autosplit
    
        # Autosplit images in the 'coco128/images' directory with default train-val-test proportions
        autosplit()
    
        # Autosplit images and include only those with corresponding annotation files
        autosplit(annotated_only=True)
        ```
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file


def verify_image_label(args):
    """
    Verifies the integrity and consistency of a given image-label pair, ensuring proper format, size, and legal label values.
    
    Args:
      args (tuple): A tuple containing the image file path (str), label file path (str), and a prefix string (str) for messages.
    
    Returns:
      tuple: A tuple containing:
        - im_file (str | None): Path to the image file if verification passes, otherwise `None`.
        - lb (numpy.ndarray): Array of labels in the image.
        - shape (tuple): Shape of the image (width, height).
        - segments (list): List of label segments, if applicable.
        - nm (int): Count of missing labels.
        - nf (int): Count of found labels.
        - ne (int): Count of empty labels.
        - nc (int): Count of corrupt labels.
        - msg (str): Diagnostic message.
    
    Raises:
      AssertionError: If the image size is less than 10 pixels in any dimension or if the label format/values are invalid.
      Exception: For various fatal errors such as file reading issues, format mismatches, and unrecognized formats.
    
    Examples:
      ```python
      img_file = 'path_to_image.jpg'
      label_file = 'path_to_label.txt'
      prefix = 'Dataset:'
    
      results = verify_image_label((img_file, label_file, prefix))
      print(results)
      ```
    
    Note:
      Images with corrupt JPEG endings are automatically restored if possible. Labels are verified to ensure that they contain 5 columns, with normalized coordinates in the range [0, 1], and are free of duplicate entries.
    """
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
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
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


class HUBDatasetStats:
    """
    Class for generating HUB dataset JSON and `-hub` dataset directory.

    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally

    Usage
        from utils.dataloaders import HUBDatasetStats
        stats = HUBDatasetStats('coco128.yaml', autodownload=True)  # usage 1
        stats = HUBDatasetStats('path/to/coco128.zip')  # usage 2
        stats.get_json(save=False)
        stats.process_images()
    """

    def __init__(self, path="coco128.yaml", autodownload=False):
        """
        Initializes the `HUBDatasetStats` class for generating HUB dataset JSON and `-hub` dataset directory.
        
        Args:
            path (str): Path to `data.yaml` or `data.zip` (with `data.yaml` inside `data.zip`). Defaults to "coco128.yaml".
            autodownload (bool): If True, attempts to download the dataset if not found locally. Defaults to False.
        
        Returns:
            None
        
        Raises:
            Exception: If there is an error loading the dataset YAML file.
        
        URL:
            See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data for more details.
        
        Examples:
            ```python
            from utils.dataloaders import HUBDatasetStats
            # Usage 1: Initialize with a YAML file
            stats = HUBDatasetStats('coco128.yaml', autodownload=True)
            
            # Usage 2: Initialize with a ZIP file
            stats = HUBDatasetStats('path/to/coco128.zip')
            
            # Generate JSON stats without saving to file
            stats.get_json(save=False)
            
            # Process images for HUB
            stats.process_images()
            ```
        """
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            with open(check_yaml(yaml_path), errors="ignore") as f:
                data = yaml.safe_load(f)  # data dict
                if zipped:
                    data["path"] = data_dir
        except Exception as e:
            raise Exception("error/HUB/dataset_stats/yaml_load") from e

        check_dataset(data, autodownload)  # download dataset if missing
        self.hub_dir = Path(data["path"] + "-hub")
        self.im_dir = self.hub_dir / "images"
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {"nc": data["nc"], "names": list(data["names"].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _find_yaml(dir):
        """
        Finds and returns the path to a YAML configuration file in the specified directory, ensuring that only one file is present.
        
        Args:
            dir (Path): Directory in which to search for the YAML file.
        
        Returns:
            Path: The path to the single '.yaml' file found in the directory.
        
        Raises:
            AssertionError: If no YAML file is found, or if more than one YAML file is detected.
        
        Notes:
            The method initially looks for files at the root level before performing a recursive search. It prefers YAML files 
            that match the stem (name) of the directory.
        """
        files = list(dir.glob("*.yaml")) or list(dir.rglob("*.yaml"))  # try root level first and then recursive
        assert files, f"No *.yaml file found in {dir}"
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f"Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed"
        assert len(files) == 1, f"Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}"
        return files[0]

    def _unzip(self, path):
        """
        _Unzips a .zip file and returns a status indicating success alongside the unzipped directory and YAML file path._
        
        Args:
            path (Path): Path to the .zip file or the .yaml file.
        
        Returns:
            tuple: (bool, Path or None, Path)
                - bool: Indicates if the file was unzipped.
                - Path or None: The path to the unzipped directory if applicable, otherwise None.
                - Path: The path to the .yaml file, either unzipped or original.
        """
        if not str(path).endswith(".zip"):  # path is data.yaml
            return False, None, path
        assert Path(path).is_file(), f"Error unzipping {path}, file not found"
        unzip_file(path, path=path.parent)
        dir = path.with_suffix("")  # dataset directory == zip name
        assert dir.is_dir(), f"Error unzipping {path}, {dir} not found. path/to/abc.zip MUST unzip to path/to/abc/"
        return True, str(dir), self._find_yaml(dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f, max_dim=1920):
        """
        Generates resized versions of images for web/app viewing, using PIL or OpenCV for processing.
        
        Args:
            f (str): Path to the original image file.
            max_dim (int): Maximum dimension (height or width) for the resized image. Defaults to 1920.
        
        Returns:
            None
        
        Notes:
            Resizes the image while maintaining the aspect ratio. Uses PIL for resizing and saving the image,
            if an exception occurs with PIL, it switches to OpenCV for both resizing and saving. The resized
            images are saved in the 'images' directory within the '-hub' directory of the dataset.
        
        Examples:
            ```python
            stats = HUBDatasetStats('path/to/coco128.yaml')
            stats._hub_ops('path/to/image.png')
            ```
        
        Reference:
            See https://github.com/ultralytics/ultralytics
        """
        f_new = self.im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, "JPEG", quality=50, optimize=True)  # save
        except Exception as e:  # use OpenCV
            LOGGER.info(f"WARNING ⚠️ HUB ops PIL failure {f}: {e}")
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    def get_json(self, save=False, verbose=False):
        """
        Retrieve dataset statistics in JSON format.
        
        Args:
            save (bool): Whether to save the statistics to a JSON file. Default is False.
            verbose (bool): Whether to print the statistics to the console. Default is False.
        
        Returns:
            dict: JSON-compatible dictionary containing dataset statistics.
            
        Examples:
            ```python
            stats = HUBDatasetStats('coco128.yaml', autodownload=True)
            stats_json = stats.get_json(save=True, verbose=True)
            ```
        
        Notes:
            - This function assumes the dataset YAML file adheres to the expected format.
            - For more information on the dataset format, refer to https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
        """

        def _round(labels):
            """Rounds class labels to integers and coordinates to 4 decimal places for improved label accuracy."""
            return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

        for split in "train", "val", "test":
            if self.data.get(split) is None:
                self.stats[split] = None  # i.e. no test set
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            x = np.array(
                [
                    np.bincount(label[:, 0].astype(int), minlength=self.data["nc"])
                    for label in tqdm(dataset.labels, total=dataset.n, desc="Statistics")
                ]
            )  # shape(128x80)
            self.stats[split] = {
                "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                "image_stats": {
                    "total": dataset.n,
                    "unlabelled": int(np.all(x == 0, 1).sum()),
                    "per_class": (x > 0).sum(0).tolist(),
                },
                "labels": [{str(Path(k).name): _round(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)],
            }

        # Save, print and return
        if save:
            stats_path = self.hub_dir / "stats.json"
            print(f"Saving {stats_path.resolve()}...")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            print(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        """
        Processes images for the Ultralytics HUB by compressing and saving them to a specified directory.
        
        Args:
            None
        
        Returns:
            None
        
        Note:
            This function compresses images from the 'train', 'val', and 'test' splits and saves them to the directory specified 
            during the initialization of the `HUBDatasetStats` instance.
            
        Example:
            ```python
            stats = HUBDatasetStats('coco128.yaml', autodownload=True)
            stats.process_images()
            ```
            
            The above code initializes the `HUBDatasetStats` object with a dataset configuration and processes the images.
        
        _warning:
            This is a resource-intensive operation and will depend on the size of the dataset and the specified number of threads.
        """
        for split in "train", "val", "test":
            if self.data.get(split) is None:
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            desc = f"{split} images"
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(self._hub_ops, dataset.im_files), total=dataset.n, desc=desc):
                pass
        print(f"Done. All images saved to {self.im_dir}")
        return self.im_dir


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.

    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        """
        Initializes a YOLOv5 Classification Dataset with the option to use caching, apply augmentations, and utilize specific 
        transforms for image classification.
        
        Args:
            root (str): The root directory of the dataset.
            augment (bool): Whether to apply Albumentations augmentations.
            imgsz (int): Desired image size.
            cache (bool | str): Whether to cache images for faster loading. If 'True' or 'ram', caches in RAM. If 'disk', caches on disk.
        
        Returns:
            None
        
        Example:
            ```python
            from ultralytics import ClassificationDataset
        
            dataset = ClassificationDataset(root='path/to/dataset', augment=True, imgsz=224, cache='ram')
            ```
        """
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == "ram"
        self.cache_disk = cache == "disk"
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        """
        Fetches and preprocesses an image and its label for classification with optional caching and augmentations.
        
        Args:
            i (int): The index of the image sample to be fetched.
        
        Returns:
            tuple(torch.Tensor, int): A tuple containing the preprocessed image as a tensor and its corresponding class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"]
        else:
            sample = self.torch_transforms(im)
        return sample, j


def create_classification_dataloader(
    path, imgsz=224, batch_size=16, augment=True, cache=False, rank=-1, workers=8, shuffle=True
):
    # Returns Dataloader object to be used with YOLOv5 Classifier
    """
    Creates a DataLoader for image classification, supporting caching, augmentation, and distributed training.
    
    Args:
        path (str): Path to the dataset directory.
        imgsz (int, optional): Desired image size. Default is 224.
        batch_size (int, optional): Number of samples per batch. Default is 16.
        augment (bool, optional): Flag to enable image augmentations. Default is True.
        cache (bool | str, optional): Cache method - 'ram' for RAM, 'disk' for disk, or False for no caching. Default is False.
        rank (int, optional): Rank for distributed training. Default is -1.
        workers (int, optional): Number of worker threads for data loading. Default is 8.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
    
    Returns:
        InfiniteDataLoader: Configured InfiniteDataLoader instance for image classification.
    
    Notes:
        - Uses PyTorch's DataLoader for loading and processing datasets.
        - Supports distributed training with torch.distributed.
    
    Example:
    ```python
    from ultralytics import create_classification_dataloader
    
    dataloader = create_classification_dataloader('/path/to/dataset', imgsz=224, batch_size=32, augment=True, cache='ram')
    ```
    """
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=generator,
    )  # or DataLoader(persistent_workers=True)
