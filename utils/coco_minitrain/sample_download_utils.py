'''
Ref: https://github.com/giddyyupp/coco-minitrain
'''
import os
import json
import argparse
from random import shuffle
from pycocotools.coco import COCO
import wget
import concurrent.futures
import pathlib
import glob

from utils.coco_minitrain.dataloader import CocoDataset
from utils.coco_minitrain.sampler_utils import get_coco_object_size_info, get_coco_class_object_counts

# default area ranges defined in coco
areaRng = [32 ** 2, 96 ** 2, 1e5 ** 2]

def sample_coco(coco_path, sample_image_count):
	run_count = 10
	train_path = coco_path + '/train2017.txt'
	save_file_name = 'instances_train2017_minicoco'
	save_format = 'json'

	annotations_path = coco_path + '/annotations'
	train_annotations_path = annotations_path + '/instances_train2017.json'
	if not os.path.exists(train_annotations_path):
		cmd = 'wget https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_train2017.json -P {}'.format(annotations_path)
		os.system(cmd)
		print(train_annotations_path, 'downloaded!')

	print('sample_coco.py - sample_coco() - coco_path: ', coco_path) # debug
	dataset_train = CocoDataset(coco_path, set_name='train2017')

	# get coco class based object counts
	annot_dict = get_coco_class_object_counts(dataset_train)
	# print(f"COCO object counts in each class:\n{annot_dict}")

	# here extract object sizes.
	size_dict = get_coco_object_size_info(dataset_train)
	# print(f"COCO object counts in each class for different sizes (S,M,L):\n{size_dict}")

	# now sample!!
	imgs_best_sample = {}
	ratio_list = []
	best_diff = 1000000
	keys = []
	# get all keys in coco train set, total image count!
	for k, v in dataset_train.coco.imgToAnns.items():
		keys.append(k)

	for rr in range(run_count):
		imgs = {}

		# shuffle keys
		shuffle(keys)

		# select first N images
		# print('sample_image_count: ', type(sample_image_count))
		for i in keys[:sample_image_count]:
			imgs[i] = dataset_train.coco.imgToAnns[i]

		# now check for category based annotations
		# annot_sampled = np.zeros(90, int)
		annot_sampled = {}
		for k, v in imgs.items():
			for it in v:
				area = it['bbox'][2] * it['bbox'][3]
				cat = it['category_id']
				if area < areaRng[0]:
					kk = str(cat) + "_S"
				elif area < areaRng[1]:
					kk = str(cat) + "_M"
				else:
					kk = str(cat) + "_L"

				if kk in annot_sampled:
					annot_sampled[kk] += 1
				else:
					annot_sampled[kk] = 1
		# print(f"Sampled Annotations dict:\n {annot_sampled}")

		# calculate ratios
		ratios_obj_count = {}
		# ratios_obj_size = {}

		failed_run = False
		for k, v in size_dict.items():
			if not k in annot_sampled:
				failed_run = True
				break

			ratios_obj_count[k] = annot_sampled[k] / float(v)
		if failed_run:
			continue

		ratio_list.append(ratios_obj_count)

		min_ratio = min(ratios_obj_count.values())
		max_ratio = max(ratios_obj_count.values())

		diff = max_ratio - min_ratio

		if diff < best_diff:
			best_diff = diff
			imgs_best_sample = imgs

		print(f"Best difference:{best_diff}")

	if save_format == 'csv':
		# now write to csv file
		save_file_path = annotations_path + '/' + save_file_name + '.csv'
		csv_file = open(save_file_path, 'w')
		write_str = ""

		for k, v in imgs_best_sample.items():
			f_name = dataset_train.coco.imgs[k]['file_name']
			for ann in v:
				bbox = ann['bbox']
				class_id = ann['category_id']
				write_str = f_name + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(
					bbox[3]) + ',' + \
							str(dataset_train.labels[dataset_train.coco_labels_inverse[class_id]]) + ',' + '0' + '\n'

				csv_file.write(write_str)

		csv_file.close()
		print('\n', save_file_path, 'saved!')

	elif save_format == 'json':
		mini_coco = {}
		annots = []
		imgs = []
		# add headers like info, licenses etc.
		mini_coco["info"] = dataset_train.coco.dataset['info']
		mini_coco["licenses"] = dataset_train.coco.dataset['licenses']
		mini_coco["categories"] = dataset_train.coco.dataset['categories']

		for k, v in imgs_best_sample.items():
			f_name = dataset_train.coco.imgs[k]['file_name']
			im_id = int(f_name[:-4])
			for ann in dataset_train.coco.imgToAnns[im_id]:
				annots.append(ann)
			imgs.append(dataset_train.coco.imgs[im_id])

		mini_coco['images'] = imgs
		mini_coco['annotations'] = annots

		save_file_path = annotations_path + '/' + save_file_name + '.json'
		with open(save_file_path, 'w') as f:
			json.dump(mini_coco, f); print('\n', save_file_path, 'saved!')


def download_sampled_images(coco_path):
    output_dir = coco_path + '/images/train2017'
    if os.path.exists(output_dir):
        output_dir_ORI = output_dir + '_ORI'
        cmd = 'mv {} {}'.format(output_dir, output_dir_ORI)
        print('Preparing', output_dir, '...')
        os.system(cmd); print(cmd)
    annotation = coco_path + '/annotations/instances_train2017_minicoco.json'
    root = pathlib.Path().absolute()
    print('coco_download.py - root: ', root)

    ann_file = root / annotation
    print('coco_download.py - ann_file: ', ann_file)

    out_p = pathlib.Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not os.fsdecode(ann_file).endswith(".json"):
        assert "Only support COCO style JSON file"

    try:
        coco = COCO(os.fsdecode(ann_file))
        img_ids = list(coco.imgs.keys())

    except FileNotFoundError:
        raise

    def download_images(id):
        try:
            start_url = "http://images.cocodataset.org/train2017"
            filename = "{0:0>12d}".format(id)
            filename = filename + ".jpg"
            full_url = f"{start_url}/{filename}"
            wget.download(full_url, out=output_dir)
        except Exception as e:
            print(f"The download exception is {e}", flush=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_images, img_ids)

    # Remove the original images folder
    if os.path.exists(output_dir):
        cmd = 'rm -r {}'.format(output_dir_ORI)
        os.system(cmd); print(cmd)

def update_labels(coco_path):
    img_path = coco_path + '/images/train2017'
    print('\n\n img_path: ', img_path)

    img_id_ls = []
    for img_path in glob.glob(img_path + '/*.jpg'):
        print(img_path)
        img_id = img_path[img_path.rindex('/') + 1 : img_path.index('.jpg')]
        print(img_id)
        # e.g. 000000443453
        img_id_ls.append(img_id)

        print(len(img_id_ls)) # 25000

    print('---------------------------')

    txt_path = coco_path + '/train2017.txt'
    print(txt_path)
    txt_path_ORI = coco_path + '/train2017_ORI.txt'
    if not os.path.exists(txt_path_ORI):
        cmd = 'scp {} {}'.format(txt_path, txt_path_ORI); os.system(cmd); print(cmd)
        cmd = 'rm {}'.format(txt_path); os.system(cmd); print(cmd)
        cmd = 'touch {}'.format(txt_path); os.system(cmd); print(cmd)

    label_root_path = coco_path + '/labels'
    print(label_root_path)
    # label_root_path_ORI = coco_path + '/labels_ORI'
    label_train_folder = label_root_path + '/train2017'
    label_train_folder_ORI = label_root_path + '/train2017_ORI'
    if not os.path.exists(label_train_folder_ORI):
        cmd = 'scp -r {} {}'.format(label_train_folder, label_train_folder_ORI); os.system(cmd); print(cmd)
        cmd = 'rm -r {}'.format(label_train_folder); os.system(cmd); print(cmd)
        os.makedirs(label_train_folder)

    print('---------------------------')

    txt_file = open(txt_path, 'w')
    for img_id in img_id_ls:
        # ----------------------
        #  Update train2017.txt
        # ----------------------
        line_to_write = './images/train2017/{}.jpg\n'.format(img_id)
        txt_file.write(line_to_write)
        print(line_to_write, 'written!')

        # ---------------------------
        #  Update labels/train2017/*
        # ---------------------------
        # label_train_path = label_root_path + '/train2017/' + img_id + '.txt'
        label_train_path = label_train_folder + '/' + img_id + '.txt'
        # label_path_ORI = label_root_path_ORI + '/train2017/' + img_id + '.txt'
        label_train_path_ORI = label_train_folder_ORI + '/' + img_id + '.txt'
        # cmd = 'scp {} {}'.format(label_path_ORI, label_path); os.system(cmd); print(print)
        cmd = 'scp {} {}'.format(label_train_path_ORI, label_train_path); os.system(cmd); print(print)
        print(cmd)

    # Clean intermediate folders
    cmd = 'rm -r {}'.format(txt_path_ORI); os.system(cmd); print(cmd)
    cmd = 'rm -r {}'.format(label_train_folder_ORI); os.system(cmd); print(cmd)
