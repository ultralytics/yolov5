import time
import json
from pathlib import Path
import glob
import torch
import numpy as np

from .utils import box_iou, xywh2xyxy, ap_per_class, coco80_to_coco91_class


class EvalResult:
    # TODO: Can be simplified by dataclasses
    # TODO: Relation of varaibles should be expressed here.
    def __init__(self, p, ap, mp, mr, mAP50, mAP, nt, num_images, ap_class):
        self.p = p
        self.ap = ap
        self.mp = mp
        self.mr = mr
        self.mAP50 = mAP50
        self.mAP = mAP
        self.nt = nt
        self.num_images = num_images
        self.ap_class = ap_class


class MetricMAP:
    def __init__(self, iou_vec, num_class):
        self.iou_vec = iou_vec
        self.niou = iou_vec.numel()
        self.nc = num_class

    def eval(self, preds_list, labels_list):
        """
        Return:
            preds: shape(len_of_existing_detections, 6(xyxy, conf, cls))
            labels: shape(len_of_gt_detections, 5(class + xyxy))
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
            mp, mr, mAP50, mAP = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.nc)  # number of targets per class
        else:
            p, r, f1, mp, mr, mAP50, mAP = 0., 0., 0., 0., 0., 0., 0.
            nt = torch.zeros(1)
            ap, ap_class = [], []

        num_images = len(labels_list)
        eval_result = EvalResult(p, ap, mp, mr, mAP50, mAP, nt, num_images, ap_class)

        time_consumed = time.time() - t0
        print("mAP evaluation cost %0.2f seconds." % time_consumed)

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
        mAP, mAP50 = None, None
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
            except Exception as e:
                print('ERROR: pycocotools unable to run: %s' % e)
                raise
        return mAP, mAP50
