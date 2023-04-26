"""eval_yolo.py

This script is for evaluating mAP (accuracy) of YOLO models.
"""
import os
import sys
import json
import argparse

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import cv2
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from Processor import Processor
from utils.general import coco80_to_coco91_class

# converts 80-index (val2014) to 91-index (paper)
coco91class = coco80_to_coco91_class()

VAL_IMGS_DIR = '../coco/images/val2017'
VAL_ANNOTATIONS = '../coco/annotations/instances_val2017.json'

def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of YOLO TRT model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--imgs-dir', type=str, default=VAL_IMGS_DIR,
        help='directory of validation images [%s]' % VAL_IMGS_DIR)
    parser.add_argument(
        '--annotations', type=str, default=VAL_ANNOTATIONS,
        help='groundtruth annotations [%s]' % VAL_ANNOTATIONS)
    parser.add_argument(
        '-c', '--category-num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument(
        '-m', '--model', type=str, default='./weights/yolov5s-simple.trt',
        help=('trt model path'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '--conf-thres', type=float, default=0.001,
        help='object confidence threshold')
    parser.add_argument(
        '--iou-thres', type=float, default=0.6,
        help='IOU threshold for NMS')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)


def generate_results(processor, imgs_dir, jpgs, results_file, conf_thres, iou_thres, non_coco):
    """Run detection on each jpg and write results to file."""
    results = []

    i = 0
    for jpg in jpgs:
        i+=1
        if(i%100 == 0):
            print('Processing {} images'.format(i))
        img = cv2.imread(os.path.join(imgs_dir, jpg))
        image_id = int(jpg.split('.')[0].split('_')[-1])
        output = processor.detect(img)

        pred = processor.post_process(output, img.shape, conf_thres=conf_thres,
                                                    iou_thres=iou_thres)
        for p in pred.tolist():
            x = float(p[0])
            y = float(p[1])
            w = float(p[2] - p[0])
            h = float(p[3] - p[1])
            results.append({'image_id': image_id,
                          'category_id': coco91class[int(p[5])] if not non_coco else int(p[5]),
                          'bbox': [round(x, 3) for x in [x, y, w, h]],
                          'score': round(p[4], 5)})

    with open(results_file, 'w') as f:
        f.write(json.dumps(results, indent=4))


def main():
    args = parse_args()
    check_args(args)

    model_prefix = args.model.replace('.trt', '').split('/')[-1]
    results_file = 'weights/results_{}.json'.format(model_prefix)

    # setup processor
    processor = Processor(model=args.model, letter_box=True)

    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    generate_results(processor, args.imgs_dir, jpgs, results_file, args.conf_thres, args.iou_thres,
                     non_coco=False)

    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(args.annotations)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
