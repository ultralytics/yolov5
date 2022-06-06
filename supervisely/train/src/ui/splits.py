import supervisely_lib as sly


def init(project_info, project_meta: sly.ProjectMeta, data, state):
    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]
    data["totalImagesCount"] = project_info.items_count

    train_percent = 80
    train_count = int(project_info.items_count / 100 * train_percent)
    state["randomSplit"] = {
        "count": {
            "total": project_info.items_count,
            "train": train_count,
            "val": project_info.items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    state["splitMethod"] = "random"

    state["trainTagName"] = ""
    if project_meta.tag_metas.get("train") is not None:
        state["trainTagName"] = "train"
    state["valTagName"] = ""
    if project_meta.tag_metas.get("val") is not None:
        state["valTagName"] = "val"

    state["trainDatasets"] = []
    state["valDatasets"] = []

    state["unlabeledImages"] = "keep"
    state["untaggedImages"] = "train"


def get_train_val_sets(project_dir, state):
    split_method = state["splitMethod"]
    sly.logger.info(f"Split method for train/val is '{split_method}'")
    if split_method == "random":
        train_count = state["randomSplit"]["count"]["train"]
        val_count = state["randomSplit"]["count"]["val"]
        train_set, val_set = sly.Project.get_train_val_splits_by_count(project_dir, train_count, val_count)
        return train_set, val_set
    elif split_method == "tags":
        train_tag_name = state["trainTagName"]
        val_tag_name = state["valTagName"]
        add_untagged_to = state["untaggedImages"]
        train_set, val_set = sly.Project.get_train_val_splits_by_tag(project_dir, train_tag_name, val_tag_name, add_untagged_to)
        return train_set, val_set
    elif split_method == "datasets":
        train_datasets = state["trainDatasets"]
        val_datasets = state["valDatasets"]
        train_set, val_set = sly.Project.get_train_val_splits_by_dataset(project_dir, train_datasets, val_datasets)
        return train_set, val_set
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")