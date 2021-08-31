# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataset loaders
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset

from utils.datasets.coco import read_json_file, load_coco_annotations
from utils.datasets.error import COCODatasetError
from utils.datasets.image_cache import ImageProvider
from utils.datasets.label_cache import LabelCache, get_hash
from utils.datasets.yolo import load_image_names_from_paths, img2label_paths

DatasetEntry = Tuple[torch.Tensor, torch.Tensor, str]


def assemble_data_loader() -> None:
    pass  # TODO


class COCODataset(Dataset):
    """
    dataset
    ├── annotations.json
    ├── dataset.cache [optional]
    └── images
        ├── image-1.jpg
        ├── image-2.jpg
        └── ...
    """

    ANNOTATIONS_FILE_NAME = "annotations.json"
    IMAGES_DIRECTORY_NAME = "images"

    def __init__(self, path: str, cache_images: Optional[str] = None) -> None:
        """
        Load COCO labels along with images from provided path.

        Args:
            path: `str` - path to `dataset` root directory.
            cache_images: `Optional[str]` - flag enabling image caching. Can be equal to one of three values: `"ram"`,
                `"disc"` or `None`. `"ram"` - all images are stored in memory to enable fastest access. This may however
                result in exceeding the limit of available memory. `"disc"` - all images are stored on hard drive but in
                raw, uncompressed form. This prevents memory overflow, and offers faster access to data then regular
                image read. `None` - image caching is turned of.
        """
        self._validate_dataset_path(path=path)
        self.path = path
        self.cache_images = cache_images
        self.image_paths, self.labels = self._load_image_paths_and_labels(path=path)
        self.image_provider = ImageProvider(cache_images=cache_images, paths=self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DatasetEntry:
        image_path = self.image_paths[index]
        labels = self.labels[index]
        image = self.image_provider.get_image(path=image_path)
        return torch.from_numpy(image), labels, image_path

    @staticmethod
    def collate_fn(batch: List[DatasetEntry]) -> torch.Tensor:
        pass  # TODO:

    @staticmethod
    def _load_image_paths_and_labels(path: str) -> Tuple[List[str], List[torch.Tensor]]:
        images_path = os.path.join(path, COCODataset.IMAGES_DIRECTORY_NAME)
        annotations_path = os.path.join(path, COCODataset.ANNOTATIONS_FILE_NAME)
        coco_data = read_json_file(file_path=annotations_path)
        coco_annotations = load_coco_annotations(coco_data=coco_data)
        image_paths = [
            os.path.join(images_path, image_name)
            for image_name
            in coco_annotations.keys()
        ]
        return image_paths, list(coco_annotations.values())

    @staticmethod
    def _validate_dataset_path(path: str) -> None:
        images_path = os.path.join(path, COCODataset.IMAGES_DIRECTORY_NAME)
        annotations_path = os.path.join(path, COCODataset.ANNOTATIONS_FILE_NAME)
        if not os.path.isfile(annotations_path) or not os.path.isdir(images_path):
            raise COCODatasetError("Given path does not point to COCO dataset.")

    @staticmethod
    def resolve_cache_path() -> Path:
        pass  # TODO:


class YOLODataset(Dataset):
    """
    dataset
    ├── image_names.txt [optional]
    ├── image_names.cache [optional]
    ├── images
    │   ├── image-1.jpg
    │   ├── image-2.jpg
    │   └── ...
    └── labels
        ├── image-1.txt
        ├── image-2.txt
        └── ...
    """

    def __init__(self, path: str, cache_images: Optional[str] = None) -> None:
        """
        Load YOLO labels along with images from provided path.

        Args:
            path: `str` - path to `images` directory or to `image_names.txt` file.
            cache_images: `Optional[str]` - flag enabling image caching. Can be equal to one of three values: `"ram"`,
                `"disc"` or `None`. `"ram"` - all images are stored in memory to enable fastest access. This may however
                result in exceeding the limit of available memory. `"disc"` - all images are stored on hard drive but in
                raw, uncompressed form. This prevents memory overflow, and offers faster access to data then regular
                image read. `None` - image caching is turned of.
        """
        self.path = path
        self.cache_images = cache_images
        self.image_paths, self.labels = self._load_image_paths_and_labels(path=path)
        self.image_provider = ImageProvider(cache_images=cache_images, paths=self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DatasetEntry:
        image_path = self.image_paths[index]
        labels = self.labels[index]
        image = self.image_provider.get_image(path=image_path)
        return torch.from_numpy(image), labels, image_path

    @staticmethod
    def collate_fn(batch: List[DatasetEntry]) -> torch.Tensor:
        pass

    @staticmethod
    def _load_image_paths_and_labels(path: str) -> Tuple[List[str], List[torch.Tensor]]:
        image_paths = load_image_names_from_paths(paths=path)
        label_paths = img2label_paths(image_paths=image_paths)

        # TODO: finalize yolo labels cache plugin
        cache_path = YOLODataset.resolve_cache_path(path=path, label_paths=label_paths)
        label_cache = LabelCache.load(
            path=cache_path,
            hash=get_hash(label_paths + image_paths)
        )
        labels = [
            label_cache[image_path]
            for image_path
            in image_paths
        ]

        return image_paths, labels

    @staticmethod
    def resolve_cache_path(path: str, label_paths: List[str]) -> Path:
        path = Path(path)
        return (path if path.is_file() else Path(label_paths[0]).parent).with_suffix('.cache')


class TransformedDataset(Dataset):

    def __init__(
            self,
            source_dataset: Dataset,
            img_size: int = 640,
            batch_size: int = 16,
            augment: bool = False,
            hyp=None,
            rect=False,
            single_cls: bool = False,
            stride: int = 32,
            pad: float = 0.0
    ) -> None:
        self.source_dataset = source_dataset
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.stride = stride
        self.single_cls = single_cls
        self.pad = pad

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> DatasetEntry:
        image, labels, image_path = self.source_dataset[index]

        if self.single_cls:
            labels[:, 0] = 0

        return image, labels, image_path
