import torch
import numpy as np

from .utils import box_iou, xywh2xyxy, ap_per_class

            
class EvalResult:
    # Can be simplified by dataclasses
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
        stats = []
        for preds, labels in zip(preds_list, labels_list):
            num_gt = len(labels)
            tcls = labels[:, 0].tolist() if num_gt else []  # target class
            if preds is None:
                if num_gt:
                    stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            # Allocate possible gt labels to predictions.
            # First, assign all predictions as incorrect
            correct = torch.zeros(preds.shape[0], self.niou, dtype=torch.bool, device=labels.device)
            if num_gt:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = labels[:, 1:5]

                # Per target class
                for _cls in torch.unique(tcls_tensor):
                    pi = (_cls == preds[:, 5]).nonzero().view(-1)  # target indices
                    # Search for detections
                    if pi.shape[0]:
                        ti = (_cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                        # Prediction to target ious
                        ious, i = box_iou(preds[pi, :4], tbox[ti]).max(1)  # best ious, indices

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