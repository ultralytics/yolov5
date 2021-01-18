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


def filter_and_transform_labels(project_dir, train_split, val_split, train_classes, yolov5_format_dir):
    new_meta = _transform_non_rectangle_classes()
    project = sly.Project(project_dir, sly.OpenMode.READ)
    for dataset in project:
        for item_name in dataset:
            ann_path = dataset.get_ann_path(item_name)
            ann_json = sly.json.load_json_file(ann_path)
            ann = sly.Annotation.from_json(ann_json, project.meta)

            # for label in labels:


