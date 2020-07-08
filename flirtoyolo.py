# taken from https://github.com/prat96/FLIR_to_Yolo

from __future__ import print_function
import argparse
import glob
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help='Directory of json files containing annotations')
    parser.add_argument(
        "output_path", help='Output directory for image.txt files')
    args = parser.parse_args()
    json_files = sorted(glob.glob(os.path.join(args.path, '*.json')))

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            images = data['images']
            annotations = data['annotations']

            file_names = []
            for i in range(0, len(images)):
                file_names.append(images[i]['file_name'])

            width = 640.0
            height = 512.0

            for i in range(0, len(images)):
                converted_results = []
                for ann in annotations:
                    if ann['image_id'] == i and ann['category_id'] <= 3:
                        cat_id = int(ann['category_id'])
                        # if cat_id <= 3:
                        left, top, bbox_width, bbox_height = map(float, ann['bbox'])

                        # Yolo classes are starting from zero index
                        cat_id -= 1
                        x_center, y_center = (
                            left + bbox_width / 2, top + bbox_height / 2)

                        # darknet expects relative values wrt image width&height
                        x_rel, y_rel = (x_center / width, y_center / height)
                        w_rel, h_rel = (bbox_width / width, bbox_height / height)
                        converted_results.append(
                            (cat_id, x_rel, y_rel, w_rel, h_rel))
                image_name = images[i]['file_name']
                image_name = image_name[14:-5]
                print(image_name)
                file = open(args.output_path + str(image_name) + '.txt', 'w+')
                file.write('\n'.join('%d %.9f %.9f %.9f %.9f' % res for res in converted_results))
                file.close()