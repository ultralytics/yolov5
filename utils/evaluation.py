import time
import json
import functools
from pathlib import Path
import logging
from dataclasses import dataclass
import glob

import torch
import numpy as np
import torch.distributed as dist

from utils.utils import box_iou, xywh2xyxy, ap_per_class, coco80_to_coco91_class
from utils.utils import plot_images, output_to_target


@dataclass
class EvalResult:
    """TODO: Annotation
    """
    p: np.array
    r: np.array
    ap: np.array
    ap50: np.array
    nt: np.array
    num_images: int
    ap_class: np.array

    @property
    def mp(self):
        return np.mean(self.p)
    @property
    def mr(self):
        return np.mean(self.r)
    @property
    def mAP(self):
        return np.mean(self.ap)
    @property
    def mAP50(self):
        return np.mean(self.ap50)


class MetricMAP:
    """A mtrci
    """
    def __init__(self, iou_vec, num_class):
        self.iou_vec = iou_vec
        self.niou = iou_vec.numel()
        self.nc = num_class

    def eval(self, preds_list, labels_list):
        """
        Args:
            preds_list: list of preds, each of which is shape(len_of_existing_detections, 6(xyxy, conf, cls)).
            labels: list of labels, each of which is shape(len_of_gt_detections, 5(class + xyxy))
        """
        t0 = time.time()
        stats = []
        for preds, labels in zip(preds_list, labels_list):
            num_gt = len(labels)
            tcls = labels[:, 0].tolist() if num_gt else []  # target class
            if preds is None or not len(preds):
                if num_gt:
                    stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            # Allocate possible gt labels to predictions.
            # First, assign all predictions as incorrect
            correct = torch.zeros(preds.shape[0], self.niou, dtype=torch.bool, device=labels.device)
            if num_gt:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # Per target class
                for _cls in torch.unique(tcls_tensor):
                    pi = (_cls == preds[:, 5]).nonzero().view(-1)  # target indices
                    # Search for detections
                    if pi.shape[0]:
                        ti = (_cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                        # Prediction to target ious
                        # TODO:
                        ious, i = box_iou(preds[pi, :4], labels[:, 1:5][ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > self.iou_vec[0]).nonzero():
                            d = ti[i[j]]  # corresponding detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > self.iou_vec  # iou_thres is 1xn
                                if len(detected) == num_gt:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            # correct: shape(len_of_detections), bool. It represents which detection is considered correct.
            # conf: shape(len_of_detections)
            # pcls: shape(len_of_detections)
            # tcls: shape(len_of_gt_boxs)
            stats.append((correct.cpu(), preds[:, 4].cpu(), preds[:, 5].cpu(), tcls))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.nc)  # number of targets per class
        else:
            p, r = [], []
            nt = torch.zeros(1)
            ap, ap50, ap_class = [], [], []

        num_images = len(labels_list)
        eval_result = EvalResult(p, r, ap, ap50, nt, num_images, ap_class)

        time_consumed = time.time() - t0
        logging.debug("mAP evaluation cost %0.2f seconds." % time_consumed)
        return eval_result

    def print_result(self, r: EvalResult, names, verbose):
        ## Print Results

        # Print evaluation results.
        print(('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'))
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', r.num_images, r.nt.sum(), r.mp, r.mr, r.mAP50, r.mAP))

        # Print results per class
        if verbose and self.nc > 1 and len(r.ap_class):
            for i, c in enumerate(r.ap_class):
                print(pf % (names[c], r.num_images, r.nt[c], r.p[i], r.r[i], r.ap50[i], r.ap[i]))


@dataclass
class EvalResultCoco:
    mAP: float
    mAP50: float


class MetricCoco:
    def __init__(self, path_list):
        self.coco91class = coco80_to_coco91_class()
        self.path_list = path_list

    def eval(self, preds_list, box_rescaled_list, save_path):
        jdict = []
        for i, (preds, box_rescaled) in enumerate(zip(preds_list, box_rescaled_list)):
            if preds is None:
                continue
            # Append to pycocotools JSON dictionary
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            path = self.path_list[i]
            image_id = Path(path).stem
            for p, b in zip(preds.tolist(), box_rescaled.tolist()):
                jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                'category_id': self.coco91class[int(p[5])],
                                'bbox': [round(x, 3) for x in b],
                                'score': round(p[4], 5)})
        if len(jdict):
            print('\nCOCO mAP with pycocotools... saving %s...' %  save_path)
            with open(save_path, 'w') as file:
                json.dump(jdict, file)

            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                imgIds = [int(Path(x).stem) for x in self.path_list]
                cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
                cocoDt = cocoGt.loadRes(save_path)  # initialize COCO pred api
                cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
                cocoEval.params.imgIds = imgIds  # image IDs to evaluate
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                mAP, mAP50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
                return EvalResultCoco(mAP, mAP50)
            except Exception as e:
                # TODO: Should be a log.error
                print('ERROR: pycocotools unable to run: %s' % e)
                raise
        return None


def do_evaluation(
        infer_results,
        infer_statistics,
        model,
        dataset,
        nc,
        verbose = False,
        do_official_coco_evaluation = False,
        official_coco_evaluation_save_fname = None
    ):
    names = model.names if hasattr(model, 'names') else model.module.names
    device = next(model.parameters()).device  # get model device
    imgsz = dataset.img_size

    t0 = infer_statistics["t0"]
    t1 = infer_statistics["t1"]
    loss = infer_statistics["loss"]
    total_batch_size = infer_statistics["batch_size"]

    ##############################
    # Calculate speeds
    t = tuple(x / len(dataset) * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, total_batch_size)  # tuple
    # Speed is presently only valid in single-gpu mode.
    if verbose:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    ##############################
    # Calculate home-made mAP
    iou_vec = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    metric = MetricMAP(iou_vec, nc)
    eval_result = metric.eval(infer_results[0], infer_results[1])
    metric.print_result(eval_result, names, verbose)
    mAP = eval_result.mAP
    mAP50 = eval_result.mAP50

    ##############################
    # Save JSON, Calculate coco mAP
    if do_official_coco_evaluation:
        metric_coco = MetricCoco(dataset.img_files)
        eval_result_coco = metric_coco.eval(infer_results[0], infer_results[2], official_coco_evaluation_save_fname)
        if eval_result_coco:
            mAP = eval_result_coco.mAP
            mAP50 = eval_result_coco.mAP50

    # TODO: This seems unuse, Comment it out first.
    ## Append to text file
    #if save_txt:
    #    gn = torch.tensor(shape[0])[[1, 0, 1, 0]]  # normalization gain whwh
    #    txt_path = str(out / Path(path).stem)
    #    preds[:, :4] = scale_coords(img.shape[1:], preds[:, :4], shape[0], shape[1])  # to original
    #    for *xyxy, conf, cls in preds:
    #        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #        with open(txt_path + '.txt', 'a') as f:
    #            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

    ##############################
    # Return results
    maps = np.zeros(nc) + mAP
    for i, c in enumerate(eval_result.ap_class):
        maps[c] = eval_result.ap[i]

    return (eval_result.mp, eval_result.mr, mAP50, mAP, *(loss.cpu() / len(dataset)).tolist()), maps, t
