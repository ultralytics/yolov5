import os
import argparse
import shutil
from xml.etree import ElementTree as ET
import cv2

SUPPORTED_DATASET = ["VOC", "COCO"]


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_type",
        type=str,
        default=SUPPORTED_DATASET[0],
        choices=SUPPORTED_DATASET,
        help=f"dataset type, support {SUPPORTED_DATASET}",
    )
    args.add_argument(
        "--raw_data_path",
        type=str,
        default="resource/ships",
        help="raw data path",
    )
    return args.parse_args()


def main():
    args = get_args()
    data_type: str = args.data_type
    raw_data_path: str = args.raw_data_path

    if data_type == "VOC":
        convert_func = convert_voc_to_yolov5
    elif data_type == "COCO":
        convert_func = convert_coco_to_yolov5
    else:
        raise ValueError(f"not supported dataset type: {data_type}")

    output_path: str = f"{raw_data_path}-yolov5"
    convert_func(raw_data_path, output_path)


def _create_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def convert_voc_to_yolov5(raw_data_path: str, output_path: str):
    rawimage_dir = os.path.join(raw_data_path, "JPEGImages")
    annotaion_dir = os.path.join(raw_data_path, "Annotations")
    image_dir = os.path.join(output_path, "images")
    label_dir = os.path.join(output_path, "labels")
    visual_dir = os.path.join(output_path, "visual")
    for dir in [output_path, image_dir, label_dir, visual_dir]:
        _create_dir(dir)

    class_names = []

    train_list = []

    for class_name in os.listdir(annotaion_dir):
        class_dir = os.path.join(annotaion_dir, class_name)
        for annotaion_file_name in os.listdir(class_dir):
            if not annotaion_file_name.endswith(".xml"):
                continue
            annotaion_file = os.path.join(class_dir, annotaion_file_name)
            image_file = os.path.join(
                rawimage_dir, class_name, annotaion_file_name.replace(".xml", ".jpg")
            )
            if not os.path.exists(image_file):
                continue

            with open(annotaion_file, "r") as f:
                xml_str = f.read()
                # xml_str = xml_str.replace("utf-8", "utf8")
                root = ET.fromstring(xml_str)
                # find width and height
                size = root.find("size")
                width = size.find("width").text
                height = size.find("height").text
                # find all objects
                object_list = []
                for obj in root.iter("object"):
                    name = obj.find("name").text
                    bndbox = obj.find("bndbox")
                    xmin = bndbox.find("xmin").text
                    ymin = bndbox.find("ymin").text
                    xmax = bndbox.find("xmax").text
                    ymax = bndbox.find("ymax").text
                    xmin = float(xmin) / float(width)
                    ymin = float(ymin) / float(height)
                    xmax = float(xmax) / float(width)
                    ymax = float(ymax) / float(height)
                    if name not in class_names:
                        class_names.append(name)
                    name_id = class_names.index(name)
                    # [ classid, x_center, y_center, w, h ]
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin
                    object_list.append([name_id, x_center, y_center, w, h])
                # write to txt file
                label_file = os.path.join(
                    label_dir, annotaion_file_name.replace(".xml", ".txt")
                )

                image_file_name = os.path.basename(image_file)
                image_file_link = os.path.join(image_dir, image_file_name)
                print(image_file, image_file_link)
                
                if len(object_list) > 0:
                    with open(label_file, "a") as f:
                        for obj in object_list:
                            f.write(f"{obj[0]} {obj[1]} {obj[2]} {obj[3]} {obj[4]}\n")
                    train_list.append(image_file_link)
                # create soft link for image
                
                shutil.copyfile(image_file, image_file_link)

            if len(object_list) > 0:
                # visual bounding box
                visual_file = os.path.join(
                    visual_dir, annotaion_file_name.replace(".xml", ".jpg")
                )

                img = cv2.imread(image_file)
                for obj in object_list:
                    x_center = int(obj[1] * float(width))
                    y_center = int(obj[2] * float(height))
                    w = int(obj[3] * float(width))
                    h = int(obj[4] * float(height))
                    xmin = int(x_center - w / 2)
                    ymin = int(y_center - h / 2)
                    xmax = int(x_center + w / 2)
                    ymax = int(y_center + h / 2)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.imwrite(visual_file, img)

    with open(os.path.join(output_path, "train.txt"), "w") as f:
        for file in train_list:
            f.write(file + "\n")
    print(f"total {len(train_list)} images")

    with open(os.path.join(output_path, "data.yaml"), "w") as f:
        # f.write(f"path: {output_path}\n")
        f.write(f"train: {output_path}/train.txt\n")
        f.write(f"val: {output_path}/train.txt\n")
        f.write("names: \n")
        for index, name in enumerate(class_names):
            f.write(f"  {index}: {name}\n")


def convert_coco_to_yolov5(raw_data_path: str):
    pass


if __name__ == "__main__":
    main()
