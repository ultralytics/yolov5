import json
import os
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm
from sklearn import model_selection

from nanovare_casa_core.utils import supervisely as sly
from nanovare_casa_core.utils import constants

ROOT_YOLO_OUTPUT = "data/mojo_yolo_dataset_grey" # Always use data to store in expected folder
USE_RGB = False

# annotation classes
ANNOTATION_CLASSES_TO_ID = {"sperm": 0}

dataset_name_id = dict(map(lambda x: (x.id, x.name), sly.Api().dataset.get_list(constants.SUPERVISELY_LOCALISATION_PROJECT_ID)))


def init_dataset(filter_by_name=["2020_01_15", "2020_01_17", "2020_01_21", "2020_01_22"], dir_name="vincent_ann_540"):
    api = sly.Api()
    api.download_project(
        constants.SUPERVISELY_LOCALISATION_PROJECT_ID
    )
    image_dir = api.merge_project(
        constants.SUPERVISELY_LOCALISATION_PROJECT_ID,
        dataset_filter_id=dict(filter(lambda x: x[1] in filter_by_name, dataset_name_id.items())),
        dir_name=dir_name,
    )
    return image_dir


def convert_to_yolo(image_dir, rgb=False):
    frames_path = list(image_dir.glob("*.png"))
    train_frame_path, test_frame_path = model_selection.train_test_split(frames_path, test_size=0.25, shuffle=True, random_state=42)
    resolved_root = Path(ROOT_YOLO_OUTPUT).resolve()
    if resolved_root.is_dir():
        shutil.rmtree(resolved_root)
    discard_count = 0
    total_count = len(frames_path)

    for frame_path in tqdm(frames_path, "Converting to YOLO format"):

        annotation_path = frame_path.parents[1] / "ann" / (frame_path.name + ".json")
        if not annotation_path.is_file():                            
            annotation_data = {
                    "description": "",
                    "tags": [],
                    "size": {
                        "height": 1200,
                        "width": 1920
                    },
                    "objects": []
                }
        else:
            with annotation_path.open() as annotation_file:
                annotation_data = json.load(annotation_file)
            discard = False
            # Remove images with "To ignore" tag
            for tag in annotation_data["tags"]:
                if tag["name"] == "To ignore":
                    discard = True
                    break
            # Remove images with no objects (to remove images not tagged yet)
            if len(annotation_data["objects"]) == 0:
                discard = True

            if discard:
                discard_count += 1
                continue

        if rgb:
            frame = cv2.imread(str(frame_path), 1)
        else:
            frame = cv2.imread(str(frame_path), 0)

        convert_to_yolo_image(frame, annotation_data, frame_path.stem, train=frame_path in train_frame_path)
    print(f"Discard {discard_count} out of {total_count}")


def convert_to_yolo_image(frame, annotation_data, image_name, train=True, w=80):
    if train:
        folder = "train"
    else:
        folder = "test"
    resolved_root = Path(ROOT_YOLO_OUTPUT).resolve()
    image_path = resolved_root / "images" / folder / f"{image_name}.jpg"
    annotation_path = resolved_root / "labels" / folder / f"{image_name}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(image_path), frame)
    im_height, im_width = frame.shape[:2]
    with annotation_path.open("w") as f:
        for obj in annotation_data["objects"]:
            if obj["classTitle"] not in ANNOTATION_CLASSES_TO_ID:
                continue
            class_id = ANNOTATION_CLASSES_TO_ID[obj["classTitle"]]
            x = obj["points"]["exterior"][0][0]
            y = obj["points"]["exterior"][0][1]
            bbox = [x, y, w, w]

            # Only keep sperm for now
            if class_id == 0:
                f.write(f"{class_id} {bbox[0]/im_width:.6f} {bbox[1]/im_height:.6f} {bbox[2]/im_width:.6f} {bbox[3]/im_height:.6f}\n")


def main():
    import train
    dataset_vincent_540 = ["2020_01_15", "2020_01_17", "2020_01_21", "2020_01_22"]
    dataset_zoe_380 = ["2020_01_24", "2020_01_23", "2020_01_16"]
    dir_name_filter_dataset = {
        "dataset_vincent_540": dataset_vincent_540,
        "dataset_zoe_380": dataset_zoe_380,
        "dataset_zoe_380_vincent_540": dataset_zoe_380 + dataset_vincent_540
    }
    for dir_name, filter_by_name in dir_name_filter_dataset.items():
        image_dir = init_dataset(filter_by_name, dir_name)
        convert_to_yolo(image_dir, rgb=USE_RGB)
        train.main()
        print(f"Project name {dir_name}")
        print("=========================     END      =================================")


if __name__ == '__main__':
    main()

