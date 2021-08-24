from typing import Tuple, List

import os
import torch
from torch.utils.data import Dataset

from utils.datasets.coco import read_json_file, ANNOTATIONS_FILE_NAME, load_coco_annotations
from utils.datasets.yolo import load_image_names_from_paths

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
    └── images
        ├── image-1.jpg
        ├── image-2.jpg
        └── ...
    """

    def __init__(self, path: str, cache_images: bool) -> None:
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


class YOLODataset(Dataset):
    """
    dataset
    ├── image_names.txt
    ├── images
    │   ├── image-1.jpg
    │   ├── image-2.jpg
    │   └── ...
    └── labels
        ├── image-1.txt
        ├── image-2.txt
        └── ...
    """

    def __init__(self, path: str, cache_images: bool) -> None:
        self.path = path
        self.cache_images = cache_images
        self.image_paths = load_image_names_from_paths(paths=path)
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
