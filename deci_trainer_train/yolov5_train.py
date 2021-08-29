from deci_trainer_train.dataset import Coco128DatasetInterface
import argparse

from models.experimental import *
from models.yolo import Model
from deci_trainer.trainer import DeciModel
from torch.optim import SGD
from loss import ComputeLoss
from deci_trainer.trainer.metrics import DetectionMetrics
from deci_trainer.trainer.models.yolov5 import YoloV5PostPredictionCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='/home/shay.aharon/PycharmProjects/yolov5/models/yolov5s.yaml',
                        help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_args()

    net = Model(opt.cfg)
    ul_hyp = {'lr0': 0.01,
              'lrf': 0.2,
              'momentum': 0.937,
              'weight_decay': 0.0005,
              'warmup_epochs': 3.0,
              'warmup_momentum': 0.8,
              'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'iou_t': 0.2,
              'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0,
              'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
              'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0}

    net.hyp = ul_hyp

    deci_model = DeciModel("yolov5_ul")
    deci_model.build_model(net, arch_params={'num_classes': 80})

    di = Coco128DatasetInterface(net, ul_hyp)
    criterion = ComputeLoss(net)
    deci_model.connect_dataset_interface(di)

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in net.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    optimizer = SGD(g0, lr=0.01, momentum=0.937, nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': 0.0005})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})

    del g0, g1, g2

    training_params = {"max_epochs": opt.epochs,
                       "lr_mode": "cosine",
                       "initial_lr": 0.01,
                       "cosine_final_lr_ratio": 0.2,
                       "lr_warmup_epochs": 3,
                       "batch_accumulate": 1,
                       "warmup_bias_lr": 0.1,
                       "loss": criterion,
                       "optimizer": optimizer,
                       "warmup_momentum": 0.8,
                       "mixed_precision": False,
                       "ema": True,
                       "train_metrics_list": [],
                       "valid_metrics_list": [DetectionMetrics(num_cls=80,
                                                               post_prediction_callback=YoloV5PostPredictionCallback())],
                       "loss_logging_items_names": ["lbox", "lobj", "lcls", "Loss"],
                       "metric_to_watch": "Loss",
                       "greater_metric_to_watch_is_better": False}

    deci_model.train(training_params=training_params)


if __name__ == "__main__":
    main()
