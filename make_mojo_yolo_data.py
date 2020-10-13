import json
from pathlib import Path
import shutil
import cv2

ROOT = r"D:\Nanovare\data\supervisely\NanovareAnnotationData\LocalizationAnnotationData\all"
ROOT_YOLO_OUTPUT = "data/mojo_yolo_dataset_kings_v2_grey" # Always use data to store in expected folder
USE_RGB = False

# annotation classes
ANNOTATION_CLASSES_TO_ID = {"sperm": 0}


def make_yolo_dataset(rgb=False):
    frames_path = Path(ROOT).resolve().glob("**/*.png")
    
    resolved_root = Path(ROOT_YOLO_OUTPUT).resolve()
    if resolved_root.is_dir():
        shutil.rmtree(resolved_root)

    for frame_path in frames_path:
        print(frame_path)

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
                print("Discarding")
                continue

        if rgb:
            frame = cv2.imread(str(frame_path), 1)
        else:
            frame = cv2.imread(str(frame_path), 0)

        convert_to_yolo_supervisely(frame, annotation_data, frame_path.stem)


def convert_to_yolo_supervisely(frame, annotation_data, image_name, w=80):
    if "cover0" in image_name or "cover2" in image_name:
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
  

if __name__ == "__main__":
    make_yolo_dataset(USE_RGB)
