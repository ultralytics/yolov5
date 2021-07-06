import os
from xml.etree.ElementTree import dump
import json
import pprint
import shutil
import tempfile
import argparse

#python3 xml2yolo.py --datasets VOC --img_path ../../kangaroo/images --label ../../kangaroo/annots --convert_output_path ../../kangaroo/annots_yolo --img_type ".jpg" --manifest_path ./ --cls_list_file ../data/yaml/kangaroo.names

from Format import VOC, COCO, UDACITY, KITTI, YOLO

parser = argparse.ArgumentParser(description='label Converting example.')

parser.add_argument('--datasets', type=str, help='type of datasets')
parser.add_argument('--img_path', type=str, help='directory of image folder')
parser.add_argument('--label', type=str,
                    help='directory of label folder or label file path')
parser.add_argument('--convert_output_path', type=str,
                    help='directory of label folder')
parser.add_argument('--img_type', type=str, help='type of image')
parser.add_argument('--manifest_path', type=str,
                    help='directory of manifest file', default="./")
parser.add_argument('--cls_list_file', type=str,
                    help='directory of *.names file', default="./")

args = parser.parse_args()

def main(config):
    if config["datasets"] == "VOC":
        voc = VOC()
        yolo = YOLO(os.path.abspath(config["cls_list"]))
        print('parsing...')

        flag, data = voc.parse(config["label"])
        if flag == True:
            print('parsing succeeded')
            flag, data = yolo.generate(data)
            if flag == True:
                print('saving results')
                if not os.path.exists(config["output_path"]):
                    os.makedirs(config["output_path"])
                flag, data = yolo.save(data,config["output_path"], config["img_path"], config["img_type"], config["manifest_path"])
            
                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("VOC Parsing Result : {}, msg : {}".format(flag, data))   

if __name__ == '__main__':

    config = {
        "datasets": args.datasets,
        "img_path": args.img_path,
        "label": args.label,
        "img_type": args.img_type,
        "manifest_path": args.manifest_path,
        "output_path": args.convert_output_path,
        "cls_list": args.cls_list_file,
    }

    main(config)
