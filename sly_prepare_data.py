import supervisely_lib as sly


def filter_and_transform_labels(project_dir):
    project = sly.Project(project_dir, sly.OpenMode.READ)
    for dataset in project:
        for item_name in dataset:
            ann_path = dataset.get_ann_path(item_name)
            ann_json = sly.json.load_json_file(ann_path)
            ann = sly.Annotation.from_json(ann_json, project.meta)

            # for label in labels:


