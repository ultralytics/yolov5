from typing import Tuple, List, Optional

import os
import torch
from torch.utils.data import Dataset

from utils.datasets.coco import read_json_file, ANNOTATIONS_FILE_NAME, load_coco_annotations
from utils.datasets.yolo import load_image_names_from_paths, img2label_paths

DatasetEntry = Tuple[torch.Tensor, torch.Tensor, str]


def assemble_data_loader() -> None:
    pass


def initiate_dataset(path: str, cache_images: bool) -> Dataset:
    if COCODataset.validate_directory_structure(path=path):
        return COCODataset(path=path, cache_images=cache_images)
    if YOLODataset.validate_directory_structure(path=path):
        return YOLODataset(path=path, cache_images=cache_images)


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

    def __init__(self, path: str, cache_images: bool) -> None:
        """
        Load COCO labels along with images from provided path.

        Args:
            path: `str` - path to `dataset` root directory.
            cache_images: `bool` - flag to force caching of images.
        """
        self.path = path
        self.cache_images = cache_images

        coco_data = read_json_file(os.path.join(path, ANNOTATIONS_FILE_NAME))
        coco_annotations = load_coco_annotations(coco_data=coco_data)

        self.image_paths = coco_annotations.keys()
        self.labels = coco_annotations.values()
        self.images = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DatasetEntry:
        pass

    @staticmethod
    def collate_fn(batch: List[DatasetEntry]) -> torch.Tensor:
        pass

    @staticmethod
    def validate_directory_structure(path: str) -> None:
        pass

    @staticmethod
    def load_labels(path: str) -> List[torch.Tensor]:
        pass

    @staticmethod
    def resolve_cache_path() -> Optional[str]:
        pass


class YOLODataset(Dataset):
    """
    dataset
    ├── image_names.txt [optional]
    ├── images
    │   ├── image-1.jpg
    │   ├── image-2.jpg
    │   └── ...
    └── labels
        ├── dataset.cache [optional]
        ├── image-1.txt
        ├── image-2.txt
        └── ...
    """

    def __init__(self, path: str, cache_images: bool) -> None:
        """
        Load YOLO labels along with images from provided path.

        Args:
            path: `str` - path to `dataset` root directory or to `image_names.txt` file.
            cache_images: `bool` - flag to force caching of images.
        """
        self.path = path
        self.cache_images = cache_images
        self.image_paths = load_image_names_from_paths(paths=path)
        self.label_paths = img2label_paths(image_paths=self.image_paths)
        self.labels = []
        self.images = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DatasetEntry:
        pass

    @staticmethod
    def collate_fn(batch: List[DatasetEntry]) -> torch.Tensor:
        pass

    @staticmethod
    def validate_directory_structure(path: str) -> None:
        pass

    @staticmethod
    def load_labels(path: str) -> List[torch.Tensor]:
        pass

    @staticmethod
    def resolve_cache_path() -> Optional[str]:
        pass


class TransformedDataset(Dataset):

    def __init__(self, source_dataset: Dataset) -> None:
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> DatasetEntry:
        return self.source_dataset[index]
