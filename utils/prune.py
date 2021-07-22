"""Prune a YOLOv5s model on a custom dataset, now only support yolov5s.yaml with one gpu, no tb or wandb in this class.
   test with GlobalWheat2020,
   with 80% GFLOPs, the prune model hold almost the same performance.
   with 5% GFLOPs, the prune model still get >90% map@.5

Usage:
    $ python3 path/to/v5/train.py--img-size 416 --batch 64 --epochs 200 --data GlobalWheat2020.yaml --cfg models/yolov5s.yaml
    $ python path/to/v5/utils/prune.py
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import yaml
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].parents[0].as_posix())  # add yolov5/ to path

import test  # for end-of-epoch mAP
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, get_latest_run, check_dataset, check_img_size, \
    set_logging, colorstr
from utils.loss import ComputeLoss
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, de_parallel
from utils.metrics import fitness


class MyPrune:
    def __init__(self, layer_filter_thresh=None, prune_num_per_iter=None, tune_epochs_pre_iter=None):
        self.logger = logging.getLogger(__name__)
        set_logging(-1)

        # # resume opt and hyp
        ckpt = get_latest_run()
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            self.opt = argparse.Namespace(**yaml.safe_load(f))
        print(self.opt)
        with open(Path(ckpt).parent.parent / 'hyp.yaml') as f:
            self.hyp = yaml.safe_load(f)
        print(self.hyp)

        self.device = select_device(self.opt.device, batch_size=self.opt.batch_size)

        self.resume_project()

        # after pruning one layer, some other layers should change accordingly, so the tensor in/out shape could match.
        # the relation will change with model structure changed, below for yolov5s

        # "key: [val[0], val[1]]" means "cur_conv_name: [next_conv_name, parallel_conv_name]"
        #  while changing the output filter nums of cur_conv, also need to change the input filter nums of next_conv
        #  and output filter nums of parallel_conv accordingly
        self.normal_convs_relation = {
            "model.0.conv.conv": [["model.1.conv"], []],
            "model.1.conv": [["model.2.cv1.conv", "model.2.cv2.conv"], []],

            "model.2.cv1.conv": [["model.2.cv3.conv", "model.2.m.0.cv1.conv"], ["model.2.m.0.cv2.conv"]],
            "model.2.m.0.cv1.conv": [["model.2.m.0.cv2.conv"], []],
            "model.2.cv3.conv": [["model.3.conv"], []],

            "model.3.conv": [["model.4.cv1.conv", "model.4.cv2.conv"], []],

            "model.4.cv1.conv": [
                ["model.4.m.0.cv1.conv", "model.4.m.1.cv1.conv", "model.4.m.2.cv1.conv", "model.4.cv3.conv"],
                ["model.4.m.0.cv2.conv", "model.4.m.1.cv2.conv", "model.4.m.2.cv2.conv"]],
            "model.4.m.0.cv1.conv": [["model.4.m.0.cv2.conv"], []],
            "model.4.m.1.cv1.conv": [["model.4.m.1.cv2.conv"], []],
            "model.4.m.2.cv1.conv": [["model.4.m.2.cv2.conv"], []],
            "model.4.cv3.conv": [["model.5.conv"], []],

            "model.5.conv": [["model.6.cv1.conv", "model.6.cv2.conv"], []],

            "model.6.cv1.conv": [
                ["model.6.m.0.cv1.conv", "model.6.m.1.cv1.conv", "model.6.m.2.cv1.conv", "model.6.cv3.conv"],
                ["model.6.m.0.cv2.conv", "model.6.m.1.cv2.conv", "model.6.m.2.cv2.conv"]],
            "model.6.m.0.cv1.conv": [["model.6.m.0.cv2.conv"], []],
            "model.6.m.1.cv1.conv": [["model.6.m.1.cv2.conv"], []],
            "model.6.m.2.cv1.conv": [["model.6.m.2.cv2.conv"], []],
            "model.6.cv3.conv": [["model.7.conv"], []],

            "model.7.conv": [["model.8.cv1.conv"], []],

            "model.8.cv2.conv": [["model.9.cv1.conv", "model.9.cv2.conv"], []],

            "model.9.cv1.conv": [["model.9.m.0.cv1.conv", "model.9.cv3.conv"], ["model.9.m.0.cv2.conv"]],
            "model.9.m.0.cv1.conv": [["model.9.m.0.cv2.conv"], []],
            "model.9.cv3.conv": [["model.10.conv"], []],

            "model.10.conv": [["model.13.cv1.conv", "model.13.cv2.conv"], []],

            "model.13.cv1.conv": [["model.13.m.0.cv1.conv", "model.13.cv3.conv"], ["model.13.m.0.cv2.conv"]],
            "model.13.m.0.cv1.conv": [["model.13.m.0.cv2.conv"], []],
            "model.13.cv3.conv": [["model.14.conv"], []],

            "model.14.conv": [["model.17.cv1.conv", "model.17.cv2.conv"], []],

            "model.17.cv1.conv": [["model.17.m.0.cv1.conv", "model.17.cv3.conv"], ["model.17.m.0.cv2.conv"]],
            "model.17.m.0.cv1.conv": [["model.17.m.0.cv2.conv"], []],
            "model.17.cv3.conv": [["model.18.conv", "model.24.m.0"], []],

            "model.18.conv": [["model.20.cv1.conv", "model.20.cv2.conv"], []],

            "model.20.cv1.conv": [["model.20.m.0.cv1.conv", "model.20.cv3.conv"], ["model.20.m.0.cv2.conv"]],
            "model.20.m.0.cv1.conv": [["model.20.m.0.cv2.conv"], []],
            "model.20.cv3.conv": [["model.21.conv", "model.24.m.1"], []],

            "model.21.conv": [["model.23.cv1.conv", "model.23.cv2.conv"], []],

            "model.23.cv1.conv": [["model.23.m.0.cv1.conv", "model.23.cv3.conv"], ["model.23.m.0.cv2.conv"]],
            "model.23.m.0.cv1.conv": [["model.23.m.0.cv2.conv"], []],
            "model.23.cv3.conv": [["model.24.m.2"], []],
        }
        # "key: [[val[0], val[1]],...]" means "concat_conv_in_back: [[concat_conv_in_front, concat_conv_out],...]"
        # if there is a concat structure, when the conv_in_back changed, the conv_out should change from index
        # conv_in_front.shape[0], rather than 0
        self.concat_convs_relation = {
            "model.2.cv2.conv": [["model.2.cv1.conv", "model.2.cv3.conv"]],
            "model.4.cv2.conv": [["model.4.cv1.conv", "model.4.cv3.conv"]],
            "model.6.cv2.conv": [["model.6.cv1.conv", "model.6.cv3.conv"]],
            "model.13.cv2.conv": [["model.13.cv1.conv", "model.13.cv3.conv"]],
            "model.17.cv2.conv": [["model.17.cv1.conv", "model.17.cv3.conv"]],
            "model.20.cv2.conv": [["model.20.cv1.conv", "model.20.cv3.conv"]],
            "model.23.cv2.conv": [["model.23.cv1.conv", "model.23.cv3.conv"]],
            "model.4.cv3.conv": [["model.14.conv", "model.17.cv1.conv"], ["model.14.conv", "model.17.cv2.conv"]],
            "model.6.cv3.conv": [["model.10.conv", "model.13.cv1.conv"], ["model.10.conv", "model.13.cv2.conv"]],
            "model.9.cv2.conv": [["model.9.cv1.conv", "model.9.cv3.conv"]],
            "model.10.conv": [["model.21.conv", "model.23.cv1.conv"], ["model.21.conv", "model.23.cv2.conv"]],
            "model.14.conv": [["model.18.conv", "model.20.cv1.conv"], ["model.18.conv", "model.20.cv2.conv"]],
        }
        # "key: val" means "cur_conv_name: "spp_in_conv: spp_out_conv"
        # spp_layer spreads the input x4, so when the input changed, the output should change x4
        self.spp_convs_relation = {
            "model.8.cv1.conv": "model.8.cv2.conv",
        }

        # check the key convs every iter, modify the key convs to modify the whole model
        self.check_convs = set()
        for check_dict in [self.normal_convs_relation, self.concat_convs_relation, self.spp_convs_relation]:
            for module in list(check_dict.keys()):
                self.check_convs.add(module)

        # minium filter nums for each layer
        self.layer_filter_thresh = layer_filter_thresh if layer_filter_thresh else \
            [8] * 3 + [16] * 2 + [32] * 2 + [64] * 3 + [32] * 4 + [16] * 6 + [32] * 3 + [64] * 2
        # filter prune nums for each iteration
        self.prune_num_per_iter = prune_num_per_iter if prune_num_per_iter else \
            [128] * 8 + [64] * 64 + [32] * 32 + [16] * 32 + [8] * 64
        # tune epochs after each iteration
        self.tune_epochs_pre_iter = tune_epochs_pre_iter if tune_epochs_pre_iter else [8] * 72 + [12] * 64 + [16] * 64
        self.iterations = len(self.tune_epochs_pre_iter)

    # with model structure changed ,the optimizer paras group... should change accordingly, codes from train.py
    def update_model_status(self):

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        if self.opt.adam:
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'] * self.hyp['lrf'],
                                        betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'] * self.hyp['lrf'], momentum=self.hyp['momentum'],
                                       nesterov=True)

        self.optimizer.add_param_group(
            {'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        self.logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # EMA
        self.ema = ModelEMA(self.model)

    # load model state_dict, codes from train.py
    def resume_project(self):
        save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, notest, nosave, workers, = \
            self.opt.save_dir, self.opt.epochs, self.opt.batch_size, self.opt.weights, self.opt.single_cls, self.opt.evolve, \
            self.opt.data, self.opt.cfg, self.opt.resume, self.opt.notest, self.opt.nosave, self.opt.workers

        with open(data) as f:
            self.data_dict = yaml.safe_load(f)  # data dict
        self.nc = 1 if single_cls else int(self.data_dict['nc'])  # number of classes
        names = ['item'] if single_cls and len(self.data_dict['names']) != 1 else self.data_dict['names']  # class names
        assert len(names) == self.nc, '%g names found for self.nc=%g dataset in %s' % (
            len(names), self.nc, data)  # check

        ckpt = torch.load(weights, map_location=self.device)  # load checkpoint
        self.model = Model(cfg or ckpt['model'].yaml, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(
            self.device)  # create
        exclude = ['anchor'] if (cfg or self.hyp.get('anchors')) and not resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
        self.model.load_state_dict(state_dict, strict=False)  # load

        self.update_model_status()

        # Optimizer
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        # EMA
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            self.ema.updates = ckpt['updates']

        # Epochs
        self.start_epoch = epochs - 1
        del ckpt, state_dict

        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        nl = self.model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        self.imgsz, self.imgsz_test = [check_img_size(x, gs) for x in self.opt.img_size]  # verify imgsz are gs-multiples

        # Trainloader
        check_dataset(self.data_dict)
        train_path = self.data_dict['train']
        test_path = self.data_dict['val']

        self.train_loader, dataset = create_dataloader(train_path, self.imgsz, batch_size, gs, single_cls,
                                                       hyp=self.hyp, augment=True, cache=self.opt.cache_images,
                                                       rect=self.opt.rect, rank=-1,
                                                       workers=workers, image_weights=self.opt.image_weights,
                                                       quad=self.opt.quad, prefix=colorstr('train: '))
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        self.nb = len(self.train_loader)  # number of batches
        assert mlc < self.nc, 'Label class %g exceeds self.nc=%g in %s. Possible class labels are 0-%g' % (
            mlc, self.nc, data, self.nc - 1)

        self.test_loader = create_dataloader(test_path, self.imgsz_test, batch_size * 2, gs, single_cls,
                                             hyp=self.hyp, cache=self.opt.cache_images and not notest, rect=True,
                                             rank=-1, workers=workers, pad=0.5, prefix=colorstr('val: '))[0]
        check_anchors(dataset, model=self.model, thr=self.hyp['anchor_t'], imgsz=self.imgsz)
        self.model.half().float()

        # Model parameters
        self.hyp['box'] *= 3. / nl  # scale to layers
        self.hyp['cls'] *= self.nc / 80. * 3. / nl  # scale to classes and layers
        self.hyp['obj'] *= (self.imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        self.hyp['label_smoothing'] = self.opt.label_smoothing
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.class_weights = labels_to_class_weights(dataset.labels, self.nc).to(
            self.device) * self.nc  # attach class weights
        self.model.names = names

    # train and test several epochs, codes from train.py
    def fine_tune(self, iteration, epochs=10):
        self.update_model_status()
        cuda = self.device.type != 'cpu'
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scaler = amp.GradScaler(enabled=cuda)
        compute_loss = ComputeLoss(self.model)  # init loss class
        ckpt = {}

        for epoch in range(epochs):
            self.model.train()
            mloss = torch.zeros(4, device=self.device)  # mean losses
            pbar = enumerate(self.train_loader)
            self.logger.info(
                ('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
            pbar = tqdm(pbar, total=self.nb)
            self.optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:
                ni = i + self.nb * epoch
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                with amp.autocast(enabled=cuda):
                    pred = self.model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                scaler.scale(loss).backward()

                scaler.step(self.optimizer)  # optimizer.step
                scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)

                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        results, maps, _ = test.run(self.data_dict,
                                    batch_size=self.opt.batch_size * 2,
                                    imgsz=self.imgsz_test,
                                    model=self.ema.ema,
                                    single_cls=False,
                                    dataloader=self.test_loader,
                                    save_dir="test",
                                    save_json=False,
                                    verbose=self.nc < 50,
                                    plots=False,
                                    wandb_logger=None,
                                    compute_loss=compute_loss)

        fi = fitness(np.array(results).reshape(1, -1))
        ckpt = {'epoch': self.opt.epochs,
                'best_fitness': fi,
                'training_results': None,
                'model': deepcopy(de_parallel(self.model)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'wandb_id': None}

        save_dir = Path(self.opt.save_dir)
        prune_wdir = save_dir / 'prune_weights'
        prune_wdir.mkdir(parents=True, exist_ok=True)  # make dir
        cur_filter_num, _ = self.count_model_filters()
        prune_w = prune_wdir / "iter_{}_ratio_{:.3f}_fitness_{:.3f}.pt".format(iteration, cur_filter_num / self.num_total, float(fi))
        torch.save(ckpt, prune_w)
        del ckpt

        return results

    # count model filters
    def count_model_filters(self, verbose=False):
        num_total, num_min = 0, 0
        for idx, (name, module) in enumerate(self.model.named_modules()):
            if verbose:
                print(name.ljust(30, " "), type(module))
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                layer_idx = int(name.split('.')[1])
                num_total += module.weight.shape[0]
                num_min += min(self.layer_filter_thresh[layer_idx], module.weight.shape[0])
        return num_total, num_min

    # find the most unimportant n filters
    def get_prune_candidates(self, n):

        # simple select way using torch.norm
        filters_rank = []
        for idx, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, torch.nn.modules.conv.Conv2d) and name in self.check_convs:
                avg_score = torch.norm(module.weight).item()
                for i in range(module.weight.shape[0]):
                    score = torch.norm(module.weight[i]).item() / module.weight[i].numel() / avg_score
                    filters_rank.append([score, i, name, module.weight.shape[0]])

        filters_rank.sort()
        from collections import defaultdict
        res = defaultdict(list)
        for score, idx, name, cnt in filters_rank:
            if n > 0 and cnt - len(res[name]) > self.layer_filter_thresh[int(name.split('.')[1])]:
                res[name].append(idx)
                n -= 1
        return res

    # auto prune layers and fine_tune
    def prune(self, verbose=False):
        self.num_total, _ = self.count_model_filters(verbose=verbose)

        from utils.torch_utils import model_info
        model_info(self.model, verbose=verbose, img_size=self.imgsz)
        self.fine_tune(iteration=0, epochs=1)

        last_prune_ratio = 1
        for iteration in range(self.iterations):
            self.model.eval()
            prune_targets = self.get_prune_candidates(self.prune_num_per_iter[iteration])
            print("*" * 30 + '\n' + "iter :{}/{} \nprune targets: {}".format(iteration, self.iterations, prune_targets))

            for layer_name, filter_indexs in prune_targets.items():
                self.prune_one_layer(layer_name, filter_indexs)

            model_info(self.model, verbose=verbose, img_size=self.imgsz)
            if self.count_model_filters()[0] / self.num_total == last_prune_ratio:
                print("meet filters limit, early stop prune")
                break
            last_prune_ratio = self.count_model_filters()[0] / self.num_total

            self.fine_tune(iteration + 1, epochs=self.tune_epochs_pre_iter[iteration])

        # final fine_tune
        self.fine_tune(self.iterations + 1, epochs=100)

    # prune certain layer with given idxs, and handle relative layers
    def prune_one_layer(self, layer_name, filter_idxs):
        if not filter_idxs:
            return

        # crop layer's output filters
        self.crop_conv_bn(layer_name, filter_idxs, crop_type="crop_output", start_idx=0)

        # handle next_layer or parallel_layer accordingly
        if layer_name in self.normal_convs_relation:
            next_layer_names = self.normal_convs_relation[layer_name][0]
            parallel_layer_names = self.normal_convs_relation[layer_name][1]

            for parallel_layer_name in parallel_layer_names:
                self.crop_conv_bn(parallel_layer_name, filter_idxs, crop_type="crop_output", start_idx=0)
            for next_layer_name in next_layer_names:
                self.crop_conv_bn(next_layer_name, filter_idxs, crop_type="crop_input", start_idx=0)

        # handle conv-concat-conv layer
        if layer_name in self.concat_convs_relation:
            for concat_idx in range(len(self.concat_convs_relation[layer_name])):
                concat_layer_name = self.concat_convs_relation[layer_name][concat_idx][0]
                next_layer_name = self.concat_convs_relation[layer_name][concat_idx][1]

                concat_layer_name = concat_layer_name.split('.')
                cur_module = self.model
                for m_name in concat_layer_name[:-1]:
                    cur_module = cur_module._modules[m_name]
                concat_layer = cur_module._modules[concat_layer_name[-1]]
                start_idx = concat_layer.weight.data.cpu().numpy().shape[0]  # crop from this idx

                self.crop_conv_bn(next_layer_name, filter_idxs, crop_type="crop_input", start_idx=start_idx)

        # handle spp layer
        if layer_name in self.spp_convs_relation:
            spp_out_layer_name = self.spp_convs_relation[layer_name]

            cur_module = self.model
            for m_name in spp_out_layer_name.split('.')[:-1]:
                cur_module = cur_module._modules[m_name]
            spp_out_layer = cur_module._modules[spp_out_layer_name.split('.')[-1]]
            spp_channel = spp_out_layer.weight.data.cpu().numpy().shape[1] // 4

            for i in range(4):
                self.crop_conv_bn(spp_out_layer_name, filter_idxs, crop_type="crop_input",
                                  start_idx=spp_channel * i - i * len(filter_idxs))

    # prune certain conv layer
    def crop_conv_bn(self, layer_name, filter_idxs, crop_type="crop_output", start_idx=0):
        # find conv
        cur_layer_name = layer_name.split('.')
        cur_module = self.model
        for m_name in cur_layer_name[:-1]:
            cur_module = cur_module._modules[m_name]
        cur_conv = cur_module._modules[cur_layer_name[-1]]

        # prune conv weights
        if crop_type == "crop_output":
            new_conv = torch.nn.Conv2d(cur_conv.in_channels, cur_conv.out_channels - len(filter_idxs),
                                       kernel_size=cur_conv.kernel_size, stride=cur_conv.stride,
                                       padding=cur_conv.padding, dilation=cur_conv.dilation,
                                       bias=(cur_conv.bias is not None))
        elif crop_type == "crop_input":
            new_conv = torch.nn.Conv2d(cur_conv.in_channels - len(filter_idxs), cur_conv.out_channels,
                                       kernel_size=cur_conv.kernel_size, stride=cur_conv.stride,
                                       padding=cur_conv.padding, dilation=cur_conv.dilation,
                                       bias=(cur_conv.bias is not None))
        else:
            raise NotImplementedError("not implemented!")

        old_conv_weights, new_conv_weights = cur_conv.weight.data.cpu().numpy(), new_conv.weight.data.cpu().numpy()
        i2 = 0  # new_conv_weights cur idx
        if crop_type == "crop_output":
            for i1 in range(old_conv_weights.shape[0]):
                if i1 not in filter_idxs:
                    new_conv_weights[i2, :, :, :] = old_conv_weights[i1, :, :, :]
                    i2 += 1
        elif crop_type == "crop_input":
            for i1 in range(old_conv_weights.shape[1] - start_idx):
                if i1 not in filter_idxs:
                    new_conv_weights[:, start_idx + i2, :, :] = old_conv_weights[:, i1, :, :]
                    i2 += 1
        else:
            raise NotImplementedError("not implemented!")
        new_conv.weight.data = torch.from_numpy(new_conv_weights).cuda()

        # prune bias weights
        if cur_conv.bias is not None:
            old_conv_bias = cur_conv.bias.data.cpu().numpy()
            new_conv_bias = new_conv.bias.data.cpu().numpy()
            i2 = 0
            for i1 in range(old_conv_weights.shape[0]):
                if i1 not in filter_idxs:
                    new_conv_bias[i2] = old_conv_bias[i1]
                    i2 += 1
            new_conv.bias.data = torch.from_numpy(new_conv_bias).cuda()

        # copy conv weights
        cur_module._modules[cur_layer_name[-1]] = new_conv

        # prune bn layer
        if crop_type == "crop_output":
            cur_bn = cur_module._modules[cur_layer_name[-1].replace("conv", "bn")]
            new_bn = torch.nn.BatchNorm2d(num_features=cur_conv.out_channels - len(filter_idxs))

            old_bn_mean, old_bn_var, old_bn_weight, old_bn_bias = cur_bn.running_mean.data.cpu().numpy(), \
                                                                  cur_bn.running_var.data.cpu().numpy(), cur_bn.weight.data.cpu().numpy(), cur_bn.bias.data.cpu().numpy()
            new_bn_mean, new_bn_var, new_bn_weight, new_bn_bias = new_bn.running_mean.data.cpu().numpy(), \
                                                                  new_bn.running_var.data.cpu().numpy(), new_bn.weight.data.cpu().numpy(), new_bn.bias.data.cpu().numpy()
            i2 = 0

            for i1 in range(old_conv_weights.shape[0]):
                if i1 not in filter_idxs:
                    new_bn_mean[i2] = old_bn_mean[i1]
                    new_bn_var[i2] = old_bn_var[i1]
                    new_bn_weight[i2] = old_bn_weight[i1]
                    new_bn_bias[i2] = old_bn_bias[i1]
                    i2 += 1
            new_bn.running_mean.data = torch.from_numpy(new_bn_mean).cuda()
            new_bn.running_var.data = torch.from_numpy(new_bn_var).cuda()
            new_bn.weight.data = torch.from_numpy(new_bn_weight).cuda()
            new_bn.bias.data = torch.from_numpy(new_bn_bias).cuda()

            cur_module._modules[cur_layer_name[-1].replace("conv", "bn")] = new_bn


if __name__ == "__main__":
    prune = MyPrune()
    prune.prune()
