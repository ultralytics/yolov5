import json
from typing import Dict, Union, List

import torch


def read_json_file(file_path: str, **kwargs) -> Union[list, dict]:
    with open(file_path, 'r') as file:
        return json.load(file, **kwargs)


def load_coco_annotations(coco_data: dict) -> Dict[str, torch.Tensor]:
    coco_image_entries_map = map_coco_image_entries(coco_image_entries=coco_data["images"])
    coco_annotation_entries_map = map_coco_annotation_entries(coco_annotation_entries=coco_data["annotations"])
    return {
        coco_image_entries_map[image_id]["file_name"]: process_coco_annotation(
            coco_annotation_entries=coco_annotation_entries_map[image_id],
            coco_image_data=coco_image_entries_map[image_id]
        )
        for image_id
        in sorted(coco_image_entries_map.keys())
    }


def map_coco_image_entries(coco_image_entries: List[dict]) -> Dict[int, dict]:
    return {
        image_data["id"]: image_data
        for image_data
        in coco_image_entries
    }


def map_coco_annotation_entries(coco_annotation_entries: List[dict]) -> Dict[int, List[dict]]:
    pass


def process_coco_annotation(coco_annotation_entries: List[dict], coco_image_data: dict) -> torch.Tensor:
    image_width = coco_image_data["width"]
    image_height = coco_image_data["height"]

