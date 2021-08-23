from typing import Tuple, List

import torch
from torch.utils.data import Dataset


DatasetEntry = Tuple[torch.Tensor, torch.Tensor, str]


def assemble_data_loader() -> None:
    pass


def initiate_dataset(path: str, cache_images: bool) -> Dataset:
    if COCODataset.validate_directory_structure(path=path):
        return COCODataset(path=path, cache_images=cache_images)
    if YOLODataset.validate_directory_structure(path=path):
        return YOLODataset(path=path, cache_images=cache_images)


class COCODataset(Dataset):

    def __init__(self, path: str, cache_images: bool) -> None:
        self.path = path
        self.cache_images = cache_images
        self.image_file_name = []

    def __len__(self) -> int:
        pass

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

    def __init__(self, path: str, cache_images: bool) -> None:
        self.path = path
        self.cache_images = cache_images
        self.image_file_name = []

    def __len__(self) -> int:
        pass

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
