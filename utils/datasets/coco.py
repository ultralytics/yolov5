import json
from typing import Dict, Union, List
from collections import defaultdict

import torch
import numpy as np


IMAGE_KEY = "images"
IMAGE_FILE_NAME_KEY = "file_name"
IMAGE_ID_KEY = "id"
IMAGE_WIDTH_KEY = "width"
IMAGE_HEIGHT_KEY = "height"
ANNOTATION_KEY = "annotations"
ANNOTATION_IMAGE_ID_KEY = "image_id"
ANNOTATION_BBOX_KEY = "bbox"
ANNOTATION_CATEGORY_ID = "category_id"


def read_json_file(file_path: str, **kwargs) -> Union[list, dict]:
    with open(file_path, 'r') as file:
        return json.load(file, **kwargs)


def load_coco_annotations(coco_data: dict) -> Dict[str, torch.Tensor]:
    coco_image_entries_map = map_coco_image_entries(coco_image_entries=coco_data[IMAGE_KEY])
    coco_annotation_entries_map = map_coco_annotation_entries(coco_annotation_entries=coco_data[ANNOTATION_KEY])
    return {
        coco_image_entries_map[image_id][IMAGE_FILE_NAME_KEY]: process_coco_annotation(
            coco_annotation_entries=coco_annotation_entries_map[image_id],
            coco_image_data=coco_image_entries_map[image_id]
        )
        for image_id
        in sorted(coco_image_entries_map.keys())
    }


def map_coco_image_entries(coco_image_entries: List[dict]) -> Dict[int, dict]:
    return {
        image_data[IMAGE_ID_KEY]: image_data
        for image_data
        in coco_image_entries
    }


def map_coco_annotation_entries(coco_annotation_entries: List[dict]) -> Dict[int, List[dict]]:
    result = defaultdict(list)
    for coco_annotation_entry in coco_annotation_entries:
        image_id = coco_annotation_entry[ANNOTATION_IMAGE_ID_KEY]
        result[image_id].append(coco_annotation_entry)
    return result


def process_coco_annotation(coco_annotation_entries: List[dict], coco_image_data: dict) -> torch.Tensor:
    image_width = coco_image_data[IMAGE_WIDTH_KEY]
    image_height = coco_image_data[IMAGE_HEIGHT_KEY]
    annotations = []
    for coco_annotation_entry in coco_annotation_entries:
        category_id = coco_annotation_entry[ANNOTATION_CATEGORY_ID]
        x_min, y_min, width, height = coco_annotation_entry[ANNOTATION_BBOX_KEY]
        annotations.append([
            0,
            category_id,
            (x_min + width / 2) / image_width,
            (y_min + height / 2) / image_height,
            width / image_width,
            height / image_height
        ])
    return torch.as_tensor(np.array(annotations))
