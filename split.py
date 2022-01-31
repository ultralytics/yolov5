import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
from loguru import logger


def train_test_split(lst_pairs: List[Any], test_size: float = 0.2, seed: int = 42):
    if 0 <= test_size <= 1:
        test_size = int(np.floor(test_size * len(lst_pairs)))
    np.random.seed(seed)
    test_idx = np.random.choice(len(lst_pairs), size=test_size, replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(lst_pairs)), test_idx)
    pairs_train = [lst_pairs[idx] for idx in train_idx]
    pairs_test = [lst_pairs[idx] for idx in test_idx]
    return pairs_train, pairs_test


def main():
    path = Path("datasets/vision2-table")
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(sorted((path / "images").glob(f"**/{ext}")))

    ## train: 0.8, val: 0.15, test: 0.05

    # first test_size is split of train and val, but we will take test from train as well, so test_size is 0.15
    image_train_paths, image_val_paths = train_test_split(image_paths, test_size=0.15)

    # second test_size we need to readjust, 0.05 * (1 - 0.15)
    image_train_paths, image_test_paths = train_test_split(image_train_paths, test_size=0.05 * (1 / (1 - 0.15)))

    (path / "images/train_split").mkdir(parents=True, exist_ok=True)
    (path / "images/val_split").mkdir(parents=True, exist_ok=True)
    (path / "images/test_split").mkdir(parents=True, exist_ok=True)
    (path / "labels/train_split").mkdir(parents=True, exist_ok=True)
    (path / "labels/val_split").mkdir(parents=True, exist_ok=True)
    (path / "labels/test_split").mkdir(parents=True, exist_ok=True)

    for image_path in image_train_paths:
        try:
            new_image_path = str(image_path.absolute()).replace("/train/", "/train_split/")
            label_path = str(image_path.absolute()).replace("/images/", "/labels/").replace(".jpeg", ".txt")
            new_label_path = label_path.replace("/train/", "/train_split/")
            shutil.move(str(image_path.absolute()), new_image_path)
            shutil.move(label_path, new_label_path)
        except Exception as e:
            logger.info(f"Exception ignored: {e}")
            continue

    for image_path in image_val_paths:
        try:
            new_image_path = str(image_path.absolute()).replace("/train/", "/val_split/")
            label_path = str(image_path.absolute()).replace("/images/", "/labels/").replace(".jpeg", ".txt")
            new_label_path = label_path.replace("/train/", "/val_split/")
            shutil.move(str(image_path.absolute()), new_image_path)
            shutil.move(label_path, new_label_path)
        except Exception as e:
            logger.info(f"Exception ignored: {e}")
            continue

    for image_path in image_test_paths:
        try:
            new_image_path = str(image_path.absolute()).replace("/train/", "/test_split/")
            label_path = str(image_path.absolute()).replace("/images/", "/labels/").replace(".jpeg", ".txt")
            new_label_path = label_path.replace("/train/", "/test_split/")
            shutil.move(str(image_path.absolute()), new_image_path)
            shutil.move(label_path, new_label_path)
        except Exception as e:
            logger.info(f"Exception ignored: {e}")
            continue


if __name__ == "__main__":
    main()
