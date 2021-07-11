import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import random
from shutil import copyfile
from typing import List
from enum import Enum

from tools.utils.general import list_files_with_extension


class DataSet(Enum):
    TRAIN = 'TRAIN'
    VAL = 'VAL'
    TEST = 'TEST'


@dataclass(frozen=True)
class DataSetSplitSpec:
    labels_source_dir_path: str
    images_source_dir_path: str
    data_set_target_dir_path: str
    train_percentage: float
    val_percentage: float
    test_percentage: float


@dataclass(frozen=True)
class Source:
    image_path: str
    label_path: str


@dataclass(frozen=True)
class SourcesSpec:
    train: List[Source]
    val: List[Source]
    test: List[Source]


@dataclass(frozen=True)
class PercentageRange:
    start: float
    end: float

    def in_range(self, value: float) -> bool:
        return self.start <= value < self.end


@dataclass(frozen=True)
class DataSetPercentageRangeSpec:
    train: PercentageRange
    val: PercentageRange
    test: PercentageRange

    @classmethod
    def from_split_spec(cls, split_spec: DataSetSplitSpec) -> 'DataSetPercentageRangeSpec':
        assert split_spec.train_percentage + split_spec.val_percentage + split_spec.test_percentage <= 1.0

        train_limit = split_spec.train_percentage
        val_limit = train_limit + split_spec.val_percentage
        test_limit = val_limit + split_spec.test_percentage

        return DataSetPercentageRangeSpec(
            train=PercentageRange(start=0.0, end=train_limit),
            val=PercentageRange(start=train_limit, end=val_limit),
            test=PercentageRange(start=val_limit, end=test_limit),
        )

    def get_subset(self) -> DataSet:
        random_value = random.uniform(0, 1)
        if self.train.in_range(value=random_value):
            return DataSet.TRAIN
        elif self.val.in_range(value=random_value):
            return DataSet.VAL
        else:
            return DataSet.TRAIN


def get_data_set_split_spec():
    parser = argparse.ArgumentParser(description='Split YOLO data set')
    parser.add_argument('--labels-source-dir-path', dest='labels_source_dir_path', type=str, required=True)
    parser.add_argument('--images-source-dir-path', dest='images_source_dir_path', type=str, required=True)
    parser.add_argument('--data-set-target-dir-path', dest='data_set_target_dir_path', type=str, required=True)
    parser.add_argument('--train-percentage', dest='train_percentage', type=float, required=True)
    parser.add_argument('--val-percentage', dest='val_percentage', type=float, required=True)
    parser.add_argument('--test-percentage', dest='test_percentage', type=float, required=True)
    args = parser.parse_args()
    return DataSetSplitSpec(
        labels_source_dir_path=args.labels_source_dir_path,
        images_source_dir_path=args.images_source_dir_path,
        data_set_target_dir_path=args.data_set_target_dir_path,
        train_percentage=args.train_percentage,
        val_percentage=args.val_percentage,
        test_percentage=args.test_percentage,
    )


def match_sources(images_dir_path: str, labels_dir_path: str) -> List[Source]:
    # label_names = list_files_with_extension(root_path=labels_dir_path, extensions='txt')
    image_names = list_files_with_extension(root_path=images_dir_path, extensions=('png', 'jpg'))
    sources = []
    for image_name in image_names:
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir_path, label_name)
        image_path = os.path.join(images_dir_path, image_name)
        if os.path.isfile(label_path) and os.path.isfile(image_path):
            sources.append(Source(
                image_path=Path(image_path).absolute().as_posix(),
                label_path=Path(label_path).absolute().as_posix()
            ))
    return sources


def split_sources(sources: List[Source], split_spec: DataSetSplitSpec) -> SourcesSpec:
    train_sources, val_sources, test_sources = [], [], []
    data_set_resolver = DataSetPercentageRangeSpec.from_split_spec(split_spec=split_spec)
    for source in sources:
        sub_set = data_set_resolver.get_subset()
        if sub_set == DataSet.TRAIN:
            train_sources.append(source)
        elif sub_set == DataSet.VAL:
            val_sources.append(source)
        else:
            test_sources.append(source)
    return SourcesSpec(
        train=train_sources,
        val=val_sources,
        test=test_sources
    )


def transfer_sources(sources: List[Source], sub_set: str, split_spec: DataSetSplitSpec) -> None:
    for source in sources:
        image_name = Path(source.image_path).name
        label_name = Path(source.label_path).name
        image_target_path = Path(split_spec.data_set_target_dir_path).joinpath('images')\
            .joinpath(sub_set).joinpath(image_name).absolute().as_posix()
        label_target_path = Path(split_spec.data_set_target_dir_path).joinpath('labels') \
            .joinpath(sub_set).joinpath(label_name).absolute().as_posix()
        Path(image_target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(label_target_path).parent.mkdir(parents=True, exist_ok=True)
        copyfile(source.image_path, image_target_path)
        copyfile(source.label_path, label_target_path)


if __name__ == "__main__":
    spec = get_data_set_split_spec()
    matched_sources = match_sources(
        images_dir_path=spec.images_source_dir_path,
        labels_dir_path=spec.labels_source_dir_path
    )
    split_out_sources = split_sources(sources=matched_sources, split_spec=spec)
    transfer_sources(sources=split_out_sources.train, sub_set='train', split_spec=spec)
    transfer_sources(sources=split_out_sources.val, sub_set='val', split_spec=spec)
    transfer_sources(sources=split_out_sources.test, sub_set='test', split_spec=spec)
