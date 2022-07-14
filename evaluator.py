# TODO:  Optimize plotting, losses & merge with val.py

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import json
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# import pycocotools.mask as mask_util
from tqdm import tqdm

from models.experimental import attempt_load
from seg_dataloaders import create_dataloader
from utils.general import (box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, )
from utils.general import (check_dataset, check_img_size, check_suffix, )
from utils.general import (coco80_to_coco91_class, increment_path, colorstr, )
from utils.plots import output_to_target, plot_images_boxes_and_masks
from utils.seg_metrics import ap_per_class, ap_per_class_box_and_mask, ConfusionMatrix
from utils.segment import (non_max_suppression_masks, mask_iou, process_mask, process_mask_upsample, scale_masks, )
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = ((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map, pred_masks=None):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

    if pred_masks is not None:
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        rles = [mask_util.encode(np.asarray(mask[:, :, None], order="F", dtype="uint8"))[0] for mask in pred_masks]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        pred_dict = {"image_id": image_id, "category_id": class_map[int(p[5])], "bbox": [round(x, 3) for x in b],
            "score": round(p[4], 5), }
        if pred_masks is not None:
            pred_dict["segmentation"] = rles[i]
        jdict.append(pred_dict)


@torch.no_grad()
class Yolov5Evaluator:
    def __init__(self, data, conf_thres=0.001, iou_thres=0.6, device="", single_cls=False, augment=False, verbose=False,
            project="runs/val", name="exp", exist_ok=False, half=True, save_dir=Path(""), nosave=False, plots=True,
            max_plot_dets=10, mask=False, mask_downsample_ratio=1, ) -> None:
        self.data = check_dataset(data)  # check
        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = iou_thres  # NMS IoU threshold
        self.device = device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.single_cls = single_cls  # treat as single-class dataset
        self.augment = augment  # augmented inference
        self.verbose = verbose  # verbose output
        self.project = project  # save to project/name
        self.name = name  # save to project/name
        self.exist_ok = exist_ok  # existing project/name ok, do not increment
        self.half = half  # use FP16 half-precision inference
        self.save_dir = save_dir
        self.nosave = nosave
        self.plots = plots
        self.max_plot_dets = max_plot_dets
        self.mask = mask
        self.mask_downsample_ratio = mask_downsample_ratio

        self.nc = 1 if self.single_cls else int(self.data["nc"])  # number of classes
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.dt = [0.0, 0.0, 0.0]
        self.names = {k: v for k, v in enumerate(self.data["names"])}
        self.s = (("%20s" + "%11s" * 10) % (
            "Class", "Images", "Labels", "Box:{P", "R", "mAP@.5", "mAP@.5:.95}", "Mask:{P", "R", "mAP@.5",
            "mAP@.5:.95}",) if self.mask else ("%20s" + "%11s" * 6) % (
            "Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95",))

        # coco stuff
        self.is_coco = isinstance(self.data.get("val"), str) and self.data["val"].endswith(
            "coco/val2017.txt")  # COCO dataset
        self.class_map = coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.jdict = []
        self.iou_thres = 0.65 if self.is_coco else self.iou_thres

        # masks stuff
        self.pred_masks = []  # for mask visualization

        # metric stuff
        self.seen = 0
        self.stats = []
        self.total_loss = torch.zeros((4 if self.mask else 3))
        self.metric = Metrics() if self.mask else Metric()

    @torch.no_grad()
    def run_training(self, model, dataloader, compute_loss=None):
        """This is for evaluation when training."""
        self.seen = 0
        self.device = next(model.parameters()).device  # get model device
        # self.iouv.to(self.device)
        self.total_loss = torch.zeros((4 if self.mask else 3), device=self.device)
        self.half &= self.device.type != "cpu"  # half precision only supported on CUDA
        model.half() if self.half else model.float()
        # Configure
        model.eval()

        # inference
        # masks will be `None` if training objection.
        for batch_i, (img, targets, paths, shapes, masks) in enumerate(tqdm(dataloader, desc=self.s)):
            # reset pred_masks
            self.pred_masks = []
            img = img.to(self.device, non_blocking=True)
            targets = targets.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            out, train_out = self.inference(model, img, targets, masks, compute_loss)

            # Statistics per image
            for si, pred in enumerate(out):
                self.seen += 1

                # eval in every image level
                labels = targets[targets[:, 0] == si, 1:]
                gt_masksi = masks[targets[:, 0] == si] if masks is not None else None

                # get predition masks
                proto_out = train_out[1][si] if isinstance(train_out, tuple) else None
                pred_maski = self.get_predmasks(pred, proto_out,
                    gt_masksi.shape[1:] if gt_masksi is not None else None, )

                # for visualization
                if self.plots and batch_i < 3 and pred_maski is not None:
                    self.pred_masks.append(pred_maski[:self.max_plot_dets].cpu())

                # NOTE: eval in training image-size space
                self.compute_stat(pred, pred_maski, labels, gt_masksi)

            if self.plots and batch_i < 3:
                import pdb;pdb.set_trace()
                self.plot_images(batch_i, img, targets, masks, out, paths)

        # compute map and print it.
        t = self.after_infer()

        # Return results
        model.float()  # for training
        return ((*self.metric.mean_results(), *(self.total_loss.cpu() / len(dataloader)).tolist(),),
                self.metric.get_maps(self.nc), t,)

    def run(self, weights, batch_size, imgsz, save_txt=False, save_conf=False, save_json=False, task="val", ):
        """This is for native evaluation."""
        model, dataloader, imgsz = self.before_infer(weights, batch_size, imgsz, save_txt, task)
        self.seen = 0
        # self.iouv.to(self.device)
        self.half &= self.device.type != "cpu"  # half precision only supported on CUDA
        model.half() if self.half else model.float()
        # Configure
        model.eval()

        # inference
        for batch_i, (img, targets, paths, shapes, masks) in enumerate(tqdm(dataloader, desc=self.s)):
            # reset pred_masks
            self.pred_masks = []
            img = img.to(self.device, non_blocking=True)
            targets = targets.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            out, train_out = self.inference(model, img, targets, masks)

            # Statistics per image
            for si, pred in enumerate(out):
                self.seen += 1
                path = Path(paths[si])
                shape = shapes[si][0]
                ratio_pad = shapes[si][1]

                # eval in every image level
                labels = targets[targets[:, 0] == si, 1:]
                gt_masksi = masks[targets[:, 0] == si] if masks is not None else None

                # get predition masks
                proto_out = train_out[1][si] if isinstance(train_out, tuple) else None
                pred_maski = self.get_predmasks(pred, proto_out,
                    gt_masksi.shape[1:] if gt_masksi is not None else None, )

                # for visualization
                if self.plots and batch_i < 3 and pred_maski is not None:
                    self.pred_masks.append(pred_maski[:self.max_plot_dets].cpu())

                # NOTE: eval in training image-size space
                self.compute_stat(pred, pred_maski, labels, gt_masksi)

                # no preditions, not save anything
                if len(pred) == 0:
                    continue

                if save_txt or save_json:
                    # clone() is for plot_images work correctly
                    predn = pred.clone()
                    # 因为test时添加了0.5的padding，因此这里与数据加载的padding不一致，所以需要转入ratio_pad
                    scale_coords(img[si].shape[1:], predn[:, :4], shape, ratio_pad)  # native-space pred

                # Save/log
                if save_txt and self.save_dir.exists():
                    # NOTE: convert coords to native space when save txt.
                    # support save box preditions only
                    save_one_txt(predn, save_conf, shape, file=self.save_dir / "labels" / (path.stem + ".txt"), )
                if save_json and self.save_dir.exists():
                    # NOTE: convert coords to native space when save json.
                    # if pred_maski is not None:
                    # h, w, n
                    pred_maski = scale_masks(img[si].shape[1:], pred_maski.permute(1, 2, 0).contiguous().cpu().numpy(),
                        shape, ratio_pad, )
                    save_one_json(predn, self.jdict, path, self.class_map,
                        pred_maski, )  # append to COCO-JSON dictionary

            if self.plots and batch_i < 3:
                self.plot_images(batch_i, img, targets, masks, out, paths)

        # compute map and print it.
        t = self.after_infer()

        # save json
        if self.save_dir.exists() and save_json:
            pred_json = str(self.save_dir / f"predictions.json")  # predictions json
            print(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
            with open(pred_json, "w") as f:
                json.dump(self.jdict, f)

        # Print speeds
        shape = (batch_size, 3, imgsz, imgsz)
        print(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

        s = (
            f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if save_txt and self.save_dir.exists() else "")
        print(f"Results saved to {colorstr('bold', self.save_dir if self.save_dir.exists() else None)}{s}")

        # Return results
        return ((*self.metric.mean_results(), *(self.total_loss.cpu() / len(dataloader)).tolist(),),
                self.metric.get_maps(self.nc), t,)

    def before_infer(self, weights, batch_size, imgsz, save_txt, task="val"):
        "prepare for evaluation without training."
        self.device = select_device(self.device, batch_size=batch_size)

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        if not self.nosave:
            (self.save_dir / "labels" if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        check_suffix(weights, ".pt")
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Data
        if self.device.type != "cpu":
            model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        pad = 0.0 if task == "speed" else 0.5
        task = (task if task in ("train", "val", "test") else "val")  # path to train/val/test images
        dataloader = create_dataloader(self.data[task], imgsz, batch_size, gs, self.single_cls, pad=pad, rect=True,
            prefix=colorstr(f"{task}: "), mask_head=self.mask, mask_downsample_ratio=self.mask_downsample_ratio, )[0]
        return model, dataloader, imgsz

    def inference(self, model, img, targets, masks=None, compute_loss=None):
        """Inference"""
        t1 = time_sync()
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Run model
        out, train_out = model(img, augment=self.augment)  # inference and training outputs
        self.dt[1] += time_sync() - t2

        # Compute loss
        if compute_loss:
            self.total_loss += compute_loss(train_out, targets, masks)[1]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(self.device)  # to pixels
        t3 = time_sync()
        out = self.nms(prediction=out, conf_thres=self.conf_thres, iou_thres=self.iou_thres, multi_label=True,
            agnostic=self.single_cls, )
        self.dt[2] += time_sync() - t3
        return out, train_out

    def after_infer(self):
        """Do something after inference, such as plots and get metrics.
        Return:
            t(tuple): speeds of per image.
        """
        # Plot confusion matrix
        if self.plots and self.save_dir.exists():
            self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        box_or_mask_any = stats[0].any() or stats[1].any()
        stats = stats[1:] if not self.mask else stats
        if len(stats) and box_or_mask_any:
            results = self.ap_per_class(*stats, self.plots, self.save_dir if self.save_dir.exists() else None,
                self.names, )
            self.metric.update(results)
            nt = np.bincount(stats[(3 if not self.mask else 4)].astype(np.int64),
                minlength=self.nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # make this empty, cause make `stats` self is for reduce some duplicated codes.
        self.stats = []
        # print information
        self.print_metric(nt, stats)
        t = tuple(x / self.seen * 1e3 for x in self.dt)  # speeds per image
        return t

    def process_batch(self, detections, labels, iouv):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(
            (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy())  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        return correct

    def get_predmasks(self, pred, proto_out, gt_shape):
        """Get pred masks in different ways.
        1. process_mask, for val when training, eval with low quality(1/mask_ratio of original size)
            mask for saving cuda memory.
        2. process_mask_upsample, for val after training to get high quality mask(original size).

        Args:
            pred(torch.Tensor): output of network, (N, 5 + mask_dim + class).
            proto_out(torch.Tensor): output of mask prototype, (mask_dim, mask_h, mask_w).
            gt_shape(tuple): shape of gt mask, this shape may not equal to input size of
                input image, Cause the mask_downsample_ratio.
        Return:
            pred_mask(torch.Tensor): predition of final masks with the same size with
                input image, (N, input_h, input_w).
        """
        if proto_out is None or len(pred) == 0:
            return None
        process = process_mask_upsample if self.plots else process_mask
        gt_shape = (gt_shape[0] * self.mask_downsample_ratio, gt_shape[1] * self.mask_downsample_ratio,)
        # n, h, w
        pred_mask = (process(proto_out, pred[:, 6:], pred[:, :4], shape=gt_shape).permute(2, 0, 1).contiguous())
        return pred_mask

    def process_batch_masks(self, predn, pred_maski, gt_masksi, labels):
        assert not ((pred_maski is None) ^ (
                    gt_masksi is None)), "`proto_out` and `gt_masksi` should be both None or both exist."
        if pred_maski is None and gt_masksi is None:
            return torch.zeros(0, self.niou, dtype=torch.bool)

        correct = torch.zeros(predn.shape[0], self.iouv.shape[0], dtype=torch.bool, device=self.iouv.device, )

        if gt_masksi.shape[1:] != pred_maski.shape[1:]:
            gt_masksi = F.interpolate(gt_masksi.unsqueeze(0), pred_maski.shape[1:], mode="bilinear",
                align_corners=False, ).squeeze(0)

        iou = mask_iou(gt_masksi.view(gt_masksi.shape[0], -1), pred_maski.view(pred_maski.shape[0], -1), )
        x = torch.where(
            (iou >= self.iouv[0]) & (labels[:, 0:1] == predn[:, 5]))  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy())  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(self.iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= self.iouv
        return correct

    def compute_stat(self, predn, pred_maski, labels, gt_maski):
        """Compute states about ious. with boxs size in training img-size space."""
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(predn) == 0:
            if nl:
                self.stats.append((torch.zeros(0, self.niou, dtype=torch.bool),  # boxes
                                   torch.zeros(0, self.niou, dtype=torch.bool),  # masks
                                   torch.Tensor(), torch.Tensor(), tcls,))
            return

        # Predictions
        if self.single_cls:
            predn[:, 5] = 0

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            # boxes
            correct_boxes = self.process_batch(predn, labelsn, self.iouv)

            # masks
            correct_masks = self.process_batch_masks(predn, pred_maski, gt_maski, labelsn)

            if self.plots:
                self.confusion_matrix.process_batch(predn, labelsn)
        else:
            correct_boxes = torch.zeros(predn.shape[0], self.niou, dtype=torch.bool)
            correct_masks = torch.zeros(predn.shape[0], self.niou, dtype=torch.bool)
        self.stats.append((correct_masks.cpu(), correct_boxes.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(),
                           tcls,))  # (correct, conf, pcls, tcls)

    def print_metric(self, nt, stats):
        # Print results
        pf = "%20s" + "%11i" * 2 + "%11.3g" * (8 if self.mask else 4)
        print(pf % ("all", self.seen, nt.sum(), *self.metric.mean_results()))

        # Print results per class
        # TODO: self.seen support verbose.
        if self.verbose and self.nc > 1 and len(stats):
            for i, c in enumerate(self.metric.ap_class_index):
                print(pf % (self.names[c], self.seen, nt[c], *self.metric.class_result(i)))

    def plot_images(self, i, img, targets, masks, out, paths):
        if not self.save_dir.exists():
            return
        # plot ground truth
        f = self.save_dir / f"val_batch{i}_labels.jpg"  # labels
        
        if masks is not None and masks.shape[1:] != img.shape[2:]:
            masks = F.interpolate(
                masks.unsqueeze(0),
                img.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        Thread(target=plot_images_boxes_and_masks, args=(img, targets, masks, paths, f, self.names, max(img.shape[2:])),
            daemon=True, ).start()
        f = self.save_dir / f"val_batch{i}_pred.jpg"  # predictions

        # plot predition
        if len(self.pred_masks):
            pred_masks = (torch.cat(self.pred_masks, dim=0) if len(self.pred_masks) > 1 else self.pred_masks[0])
        else:
            pred_masks = None
        Thread(target=plot_images_boxes_and_masks,
            args=(img, output_to_target(out, filter_dets=self.max_plot_dets), pred_masks, paths, f, self.names, max(img.shape[2:]),),
            daemon=True, ).start()
        import wandb
        if wandb.run:
            wandb.log({f"pred_{i}": wandb.Image(f)})

    def nms(self, **kwargs):
        return (non_max_suppression_masks(**kwargs) if self.mask else non_max_suppression(**kwargs))

    def ap_per_class(self, *args):
        return ap_per_class_box_and_mask(*args) if self.mask else ap_per_class(*args)


class Metric:
    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """AP@0.5 of all classes.
        Return:
            (nc, ) or [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Return:
            (nc, ) or [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """mean precision of all classes.
        Return:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """mean recall of all classes.
        Return:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """Mean AP@0.5 of all classes.
        Return:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """Mean AP@0.5:0.95 of all classes.
        Return:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map"""
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]"""
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Args:
            results: tuple(p, r, ap, f1, ap_class)
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Metric for boxes and masks."""

    def __init__(self) -> None:
        self.metric_box = Metric()
        self.metric_mask = Metric()

    def update(self, results):
        """
        Args:
            results: Dict{'boxes': Dict{}, 'masks': Dict{}}
        """
        self.metric_box.update(list(results["boxes"].values()))
        self.metric_mask.update(list(results["masks"].values()))

    def mean_results(self):
        return self.metric_box.mean_results() + self.metric_mask.mean_results()

    def class_result(self, i):
        return self.metric_box.class_result(i) + self.metric_mask.class_result(i)

    def get_maps(self, nc):
        return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)

    @property
    def ap_class_index(self):
        # boxes and masks have the same ap_class_index
        return self.metric_box.ap_class_index
