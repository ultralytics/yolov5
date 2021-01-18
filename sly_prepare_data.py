import os
import supervisely_lib as sly


def _transform_non_rectangle_classes(META: sly.ProjectMeta, train_classes):
    new_classes = []
    for class_name in train_classes:
        obj_class = META.get_obj_class(class_name)
        obj_class: sly.ObjClass
        if obj_class is None:
            raise KeyError(f"Class {class_name} not found")
        if obj_class.geometry_type == sly.Rectangle:
            new_classes.append(obj_class.clone())
        else:
            new_classes.append(obj_class.clone(geometry_type=sly.Rectangle))
    return sly.ProjectMeta(obj_classes=sly.ObjClassCollection(new_classes))


def transform_label(class_names, img_size, label: sly.Label):
    class_number = class_names.index(label.obj_class.name)
    rect_geometry = label.geometry.to_bbox()
    center = rect_geometry.center
    x_center = round(center.col / img_size[1], 6)
    y_center = round(center.row / img_size[0], 6)
    width = round(rect_geometry.width / img_size[1], 6)
    height = round(rect_geometry.height / img_size[0], 6)
    result = '{} {} {} {} {}'.format(class_number, x_center, y_center, width, height)
    return result


def _create_data_config(output_dir, meta: sly.ProjectMeta):
    class_names = [obj_class.name for obj_class in meta.obj_classes]
    class_colors = [obj_class.color for obj_class in meta.obj_classes]
    data_yaml = {
        "train": os.path.join(output_dir, "images/train"),
        "val": os.path.join(output_dir, "images/val"),
        "labels_train": os.path.join(output_dir, "labels/train"),
        "labels_val": os.path.join(output_dir, "labels/val"),
        "nc": len(class_names),
        "names": class_names,
        "colors": class_colors
    }
    sly.fs.mkdir(data_yaml["train"])
    sly.fs.mkdir(data_yaml["val"])
    sly.fs.mkdir(data_yaml["labels_train"])
    sly.fs.mkdir(data_yaml["labels_val"])
    return data_yaml


def transform_annotation(ann, class_names, save_path):
    yolov5_ann = []
    for label in ann.labels:
        yolov5_ann.append(transform_label(class_names, ann.img_size, label))
    if len(yolov5_ann) > 0:
        with open(save_path, 'w') as file:
            file.write("\n".join(yolov5_ann))
        return True
    return False


def filter_and_transform_labels(input_dir, META, train_classes,
                                train_split, val_split,
                                output_dir):
    new_meta = _transform_non_rectangle_classes(META, train_classes)
    data_yaml = _create_data_config(output_dir)

    project = sly.Project(input_dir, sly.OpenMode.READ)
    for dataset_name, item_name in train_split:
        dataset = project.datasets.get(dataset_name)
        ann_path = dataset.get_ann_path(item_name)
        ann_json = sly.json.load_json_file(ann_path)
        ann = sly.Annotation.from_json(ann_json, project.meta)

        save_ann_path = os.path.join(data_yaml["labels_train"], f"{dataset_name}_{sly.fs.get_file_name(item_name)}.txt")
        not_empty = transform_annotation(ann, data_yaml["names"], save_ann_path)
        if not_empty:
            img_path = dataset.get_img_path(item_name)
            save_img_path = os.path.join(data_yaml["train"], item_name)
            sly.fs.copy_file(img_path, save_img_path)


    #project = sly.Project(input_dir, sly.OpenMode.READ)
    # for dataset in project:
    #     for item_name in dataset:
    #         ann_path = dataset.get_ann_path(item_name)
    #         ann_json = sly.json.load_json_file(ann_path)
    #         ann = sly.Annotation.from_json(ann_json, project.meta)
    #         for label in labels:

