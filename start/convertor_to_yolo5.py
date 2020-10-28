import supervisely_lib as sly
import os

workspace_id = 23821
project_id = 103287#103276#103162#103143#102425

config_file_path = '/alex_data/sl.yaml'
train_images = 'train: /alex_data/sl/images/train/\n'
val_images = 'val: /alex_data/sl/images/train/\n'

api = sly.Api.from_env()
meta_json = api.project.get_meta(project_id)
meta = sly.ProjectMeta.from_json(meta_json)

classe_names = [obj_class.name for obj_class in meta.obj_classes]
number_of_classes = 'nc: {}\n'.format(len(classe_names))
names = 'names: {}'.format(classe_names)
#with open(config_file_path, 'a') as f:
#    f.write(train_images)
#    f.write(val_images)
#    f.write(number_of_classes)
#    f.write(names)


volo5_data_train = '/alex_data/sl/images/train'
volo5_data_val = '/alex_data/sl/images/val'

volo5_ann_train = '/alex_data/sl/labels/train'
volo5_ann_val = '/alex_data/sl/labels/val'

datasets = api.dataset.get_list(project_id)
progress = sly.Progress("Start convertion", api.project.get_images_count(project_id))
for dataset in datasets:
    images = api.image.get_list(dataset.id)
    for batch in sly.batched(images):
        image_ids = [image_info.id for image_info in batch]
        image_names = [image_info.name for image_info in batch]
        images_dir_path = [os.path.join(volo5_data_train, im) for im in image_names]
        #api.image.download_paths(dataset.id, image_ids, images_dir_path)
        ann_infos = api.annotation.download_batch(dataset.id, image_ids)
        for ann_info in ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, meta)
            for label in ann.labels:
                class_number = classe_names.index(label.obj_class.name)
                rect_geometry = label.geometry.to_bbox()
                center = rect_geometry.center
                x_center = round(center.col / ann.img_size[1], 6)
                y_center = round(center.row / ann.img_size[0], 6)
                width = round(rect_geometry.width / ann.img_size[1], 6)
                height = round(rect_geometry.height / ann.img_size[0], 6)
                res_obj_line = '{} {} {} {} {}\n'.format(class_number, x_center, y_center, width, height)
                ann_path = os.path.join(volo5_ann_train, (ann_info.image_name.split('.')[0] + '.txt'))
                with open(ann_path, 'a') as f:
                    f.write(res_obj_line)

