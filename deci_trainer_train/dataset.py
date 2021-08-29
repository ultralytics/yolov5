from deci_trainer.trainer.datasets import DatasetInterface

from utils.datasets import create_dataloader, LoadImagesAndLabels
from utils.general import colorstr
import torch
coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear',
                'hair drier', 'toothbrush']



class Coco128DatasetInterface(DatasetInterface):
    def __init__(self, model, hyp):


        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        _, trainset = create_dataloader(path='/home/shay.aharon/PycharmProjects/datasets/coco128/images/train2017',
                                        imgsz=320, batch_size=2, stride=gs, hyp=hyp, augment=True,
                                        prefix=colorstr('train: '))

        _, testset = create_dataloader(path='/home/shay.aharon/PycharmProjects/datasets/coco128/images/train2017',
                                       imgsz=320, batch_size=2, stride=gs, hyp=hyp, augment=False, pad=0.5,
                                       prefix=colorstr('val: '))

        setattr(trainset, 'classes', coco_classes)
        setattr(testset, 'classes', coco_classes)

        self.trainset = trainset
        self.testset = testset
        dataset_params = {'test_batch_size': 64,
                          "test_collate_fn": LoadImagesAndLabels.collate_fn,
                          "train_collate_fn": LoadImagesAndLabels.collate_fn}

        super(Coco128DatasetInterface, self).__init__(dataset_params)
