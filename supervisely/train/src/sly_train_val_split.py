import random
import supervisely_lib as sly


def _list_items(project_dir):
    items = []
    project = sly.Project(project_dir, sly.OpenMode.READ)
    for dataset in project:
        for item_name in dataset:
            items.append((dataset.name, item_name))
    return items


def _split_random(project_dir, train_count, val_count):
    items = _list_items(project_dir)
    random.shuffle(items)
    train_items = items[:train_count]
    val_items = items[train_count:]
    if len(val_items) != val_count:
        sly.logger.warn("Incorrect train/val random split", extra={
            "train_count": train_count,
            "val_count": val_count,
            "items_count": len(items)
        })
        raise RuntimeError("Incorrect train/val random split")
    return train_items, val_items


def _split_same(project_dir):
    items = _list_items(project_dir)
    return items, items.copy()


def _split_tags(project_dir, train_tag_name, val_tag_name):
    raise NotImplementedError()


def train_val_split(project_dir, state):
    split_method = state["splitMethod"]
    train_count = state["randomSplit"]["count"]["train"]
    val_count = state["randomSplit"]["count"]["val"]

    train_split = None
    val_split = None
    if split_method == 1:  # Random
        train_split, val_split = _split_random(project_dir, train_count, val_count)
    elif split_method == 2:  # Based on image tags
        train_split, val_split = _split_tags()
    elif split_method == 3:  # Train = Val
        train_split, val_split = _split_same()
    else:
        raise ValueError(f"Train/val split method: {split_method} unknown")

    return train_split, val_split