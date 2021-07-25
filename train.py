import argparse
import logging
import math
import os
import random
import time
import cv2
from copy import deepcopy
from pathlib import Path
from threading import Thread
from torch.autograd import Variable
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from utils.general import xywh2xyxy
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
# from models.yolo import Model
from models.yolo_mask import Model
from utils.autoanchor import check_anchors
# from utils.datasets import create_dataloader
from utils.custom_dataset import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, resume_and_get_id

logger = logging.getLogger(__name__)


def detection_target_layer(rpn_proposals, gt_class_ids, gt_boxes, gt_masks):

  if len(gt_boxes.size()) == 1:
    print('1 ground truth box')
    gt_boxes = gt_boxes.view(1, 4)
  proposals = rpn_proposals

  overlaps = bbox_overlaps(proposals, gt_boxes)

  # Determine postive and negative ROIs
  roi_iou_max = torch.max(overlaps, dim=1)[0]

  # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
  positive_roi_bool = roi_iou_max >= 0.7

  # Subsample ROIs. Aim for 33% positive
  # Positive ROIs
  if torch.nonzero(positive_roi_bool).nelement() != 0:
    positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

    # TODO: change these numbers
    positive_count = int(len(rpn_proposals) / 2)
    rand_idx = torch.randperm(positive_indices.size()[0])
    rand_idx = rand_idx[:positive_count].cuda()

    positive_indices = positive_indices[rand_idx]
    positive_count = positive_indices.size()[0]
    positive_rois = proposals[positive_indices.data, :]

    # Assign positive ROIs to GT boxes.
    positive_overlaps = overlaps[positive_indices.data, :]
    roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
    roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
    roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

    # Assign positive ROIs to GT masks
    roi_masks = gt_masks[roi_gt_box_assignment.data, :, :]

    # Transform ROI corrdinates from normalized image space
    # to normalized mini-mask space.
    x1, y1, x2, y2 = positive_rois.chunk(4, dim=1)
    gt_x1, gt_y1, gt_x2, gt_y2 = roi_gt_boxes.chunk(4, dim=1)
    gt_h = gt_y2 - gt_y1
    gt_w = gt_x2 - gt_x1
    y1 = (y1 - gt_y1) / gt_h
    x1 = (x1 - gt_x1) / gt_w
    y2 = (y2 - gt_y1) / gt_h
    x2 = (x2 - gt_x1) / gt_w
    boxes = torch.cat([y1, x1, y2, x2], dim=1) #crop and resize (y1,x1,y2,x2)

    box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
    box_ids = box_ids.cuda()

    # masks = CropAndResizeFunction(28, 28, 0)(roi_masks.unsqueeze(1), boxes, box_ids).data
    masks = Variable(CropAndResizeFunction.apply(roi_masks.unsqueeze(1), boxes, box_ids, 28, 28).data, requires_grad=False)
    masks = masks.squeeze(1)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = torch.round(masks)
  else:
    positive_count = 0

  # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
  negative_roi_bool = roi_iou_max < 0.5
  negative_roi_bool = negative_roi_bool
  # Negative ROIs. Add enough to maintain positive:negative ratio.
  if torch.nonzero(negative_roi_bool).nelement() != 0 and positive_count > 0:
    negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
    r = 1.0 / 0.33
    negative_count = int(r * positive_count - positive_count)
    rand_idx = torch.randperm(negative_indices.size()[0])
    rand_idx = rand_idx[:negative_count].cuda()

    negative_indices = negative_indices[rand_idx]
    negative_count = negative_indices.size()[0]
    negative_rois = proposals[negative_indices.data, :]
  else:
    negative_count = 0

  # Append negative ROIs and pad bbox deltas and masks that
  # are not used for negative ROIs with zeros.
  if positive_count > 0 and negative_count > 0:
    rois = torch.cat((positive_rois, negative_rois), dim=0)
    zeros = Variable(torch.zeros(negative_count), requires_grad=False).float().cuda()
    roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
    zeros = Variable(torch.zeros(negative_count, 28, 28), requires_grad=False)
    zeros = zeros.cuda()
    masks = torch.cat([masks, zeros], dim=0)
  elif positive_count > 0:
    rois = positive_rois
  elif negative_count > 0:
    rois = negative_rois
    zeros = Variable(torch.zeros(negative_count), requires_grad=False)

    zeros = zeros.cuda()
    roi_gt_class_ids = zeros
    zeros = Variable(torch.zeros(negative_count, 28, 28), requires_grad=False)

    zeros = zeros.cuda()
    masks = zeros
  else:
    rois = Variable(torch.FloatTensor(), requires_grad=False).cuda()
    roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False).cuda()
    masks = Variable(torch.FloatTensor(), requires_grad=False).cuda()

  return rois, roi_gt_class_ids, masks


def bbox_overlaps(boxes1, boxes2):
    if len(boxes1.size()) == 1:
        boxes1 = boxes1.unsqueeze(0)
    if len(boxes2.size()) == 1:
        boxes2 = boxes2.unsqueeze(0)

    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]

    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)

    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps

class MaskLoss(torch.nn.Module):
    def __init(self):
        super(MaskLoss, self).__init__()
    def forward(self, target_masks, target_class_ids, pred_masks):
        if torch.nonzero(target_class_ids > 0).nelement() != 0:
            positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
            positive_class_ids = target_class_ids[positive_ix.data].long()
            indices = torch.stack((positive_ix, positive_class_ids), dim=1)

            # Gather the masks (predicted and true) that contribute to loss
            y_true = target_masks[indices[:, 0].data, :, :]
            y_pred = pred_masks[indices[:, 0].data, indices[:, 1].data, :, :]
            if not y_true.is_cuda:
                y_true = y_true.cuda()
            if not y_pred.is_cuda:
                y_pred = y_pred.cuda()
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

            for i in range(y_true.size()[0]):
                jo = np.array(y_pred[i].cpu().detach(), dtype='float32')
                cv2.imshow('mask_predict',
                           np.hstack((cv2.resize(np.array(y_true[i].cpu().detach(), dtype='float32'), (300, 300)),
                                      cv2.resize(jo, (300, 300)))))
                cv2.waitKey(500)

        else:
            loss = torch.FloatTensor([0], requires_grad= False)

        return loss

def set_device(model, gpus, chunk_sizes, device, optimizer):
    model = model.to(device)
    if isinstance(optimizer, list):
        for opti in optimizer:
            for state in opti.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=device, non_blocking=True)
    else:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        # logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    # with torch_distributed_zero_first(rank):
    #     check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # if any(x in k for x in freeze):
        if 'mask_model' not in k:
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # if opt.adam:
    #     optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    # else:
    #     optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    #
    # optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    RPN_parameters = [param for name, param in model.model.named_parameters() if param.requires_grad]
    Mask_parameters_wo_bn = [param for name, param in model.mask_model.named_parameters() if param.requires_grad and 'bn' not in name]
    Mask_parameters_w_bn = [param for name, param in model.mask_model.named_parameters() if param.requires_grad and 'bn' in name]

    # for name, param in model.backbone_model.named_parameters():
    #     #if 'mask' not in name:
    #     print(name)
    # for name, param in model.mask_model.named_parameters():
    #     if 'bn' not in name:
    #         print(name)

    optimizer1 = torch.optim.Adam([{'params': RPN_parameters}], hyp['lr0'])
    optimizer2 = torch.optim.SGD(
        [{'params': Mask_parameters_wo_bn, 'weight_decay': 0.00005}, {'params': Mask_parameters_w_bn}]
        , lr=0.01, momentum=0.9)
    optimizer = [optimizer1, optimizer2]
    # set_device(model, 1, 1, 'cuda', optimizer)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    # ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        # if ckpt['optimizer'] is not None:
        #     optimizer.load_state_dict(ckpt['optimizer'])
        #     best_fitness = ckpt['best_fitness']

        # EMA
        # if ema and ckpt.get('ema'):
        #     ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        #     ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = 0
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    # if cuda and rank == -1 and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    print('number of batch', nb)
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        # testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
        #                                hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
        #                                world_size=opt.world_size, workers=opt.workers,
        #                                pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision


    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # scheduler.last_epoch = start_epoch - 1  # do not move

    compute_loss = ComputeLoss(model)  # init loss class
    Mask_loss = MaskLoss()
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    model.train()

    for epoch in range(start_epoch, epochs):
        batch_iter = iter(dataloader)
        imgs, targets, paths, shapes, masks = next(batch_iter)
        [z.zero_grad() for z in optimizer]

        # batch -------------------------------------------------------------
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets = targets.to(device)
        gt_class_ids = targets[:, 0] + 1

        # Multi-scale
        new_gt_masks = []
        for mask in masks:
            for m in mask:
                m = m.astype(np.float32)
                mask_image = np.zeros((imgs.shape[2], imgs.shape[3]), dtype=np.float32)
                cv2.fillPoly(mask_image, m.reshape(1, -1, 2).astype(np.int32), (1, 1, 1))
                x,y,w,h = cv2.boundingRect(m)
                crop_mask = mask_image[y:y+h, x:x+w]
                if crop_mask.shape[0] == 0 or crop_mask.shape[1] == 0:
                    continue
                crop_mask = cv2.resize(crop_mask, (28, 28))
                # cv2.imshow('ca', crop_mask)
                # cv2.waitKey()
                new_gt_masks.append(crop_mask)

        if len(new_gt_masks) != 0:
            new_gt_masks = [torch.from_numpy(i).unsqueeze(0).cuda() for i in new_gt_masks]
            new_gt_masks = torch.cat(new_gt_masks, 0)

            # Forward
            # with amp.autocast(enabled=cuda):
            pred, output_features = model(imgs)  # forward
            loss, loss_items = compute_loss(pred[1], targets.to(device))  # loss scaled by batch_size
            loss *= opt.world_size  # gradient averaged between devices in DDP mode

            ##### MASK calculation
            rpn_masks = []
            mrcnn_masks = []
            roi_gt_class_ids = []

            rpn_proposals = get_top_k(pred[2][1].clone().detach(), 100)[0]
            gt_boxes_mask = []
            for gtb in targets:
                temp = gtb.tolist()
                new_gtb = torch.FloatTensor([temp[3]-temp[5]/2, temp[2]-temp[4]/2,
                                             temp[3]+temp[5]/2, temp[2]+temp[4]/2]).cuda()
                gt_boxes_mask.append(new_gtb)

            rpn_proposals_mask = []
            for rpnm in rpn_proposals:
                temp = rpnm.tolist()
                new_rpnm = torch.FloatTensor([temp[1]/imgs.shape[3], temp[0]/imgs.shape[2],
                                              temp[3]/imgs.shape[3], temp[2]/imgs.shape[2]]).cuda()
                rpn_proposals_mask.append(new_rpnm)

            gt_boxes_mask = torch.stack(gt_boxes_mask).cuda()
            if len(rpn_proposals_mask) != 0:
                rpn_proposals_mask = torch.stack(rpn_proposals_mask).cuda()
                new_rpn_proposals = torch.cat((gt_boxes_mask, rpn_proposals_mask), dim= 0)
            else:
                new_rpn_proposals = gt_boxes_mask
            rois, roi_gt_class_id, rpn_mask = detection_target_layer(new_rpn_proposals, gt_class_ids, gt_boxes_mask,
                                                                     new_gt_masks)

            mrcnn_mask = model.mask_model([output_features[0][0].type(torch.float32)], rois, list(imgs.shape[2:]))
            rpn_masks.extend(rpn_mask)
            mrcnn_masks.extend(mrcnn_mask)
            roi_gt_class_ids.extend(roi_gt_class_id)
            roi_gt_class_ids = torch.IntTensor(roi_gt_class_ids)
            rpn_masks = torch.stack(rpn_masks)
            mrcnn_masks = torch.stack(mrcnn_masks)
            mask_loss = Mask_loss(rpn_masks, roi_gt_class_ids, mrcnn_masks)
            loss[0] += mask_loss.mean()
            loss.backward()
            [i.step() for i in optimizer]

            print('epoch: {} | loss_box: {}, loss_obj: {}, loss_cls: {}, loss_mask: {}'.format(epoch, loss_items[0].data,
                                                        loss_items[1].data, loss_items[2].data, mask_loss.data))

            if np.mod(epoch, 1000) == 0 and epoch > 0:
                save_model(os.path.join('runs/yolo_mask', 'model_{}.pth'.format(epoch)), epoch, model, optimizer=None)

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

def get_top_k(prediction, top_k=200, classes=None, labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    conf_thres = 0.5
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres

    # Settings
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > top_k:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:top_k]]  # sort by confidence

        output[xi] = x
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/best.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5x.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/custom.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_false', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_false', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_false', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=1, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Resume
    wandb_run = resume_and_get_id(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

