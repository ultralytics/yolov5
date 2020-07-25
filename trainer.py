import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *

try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    pass

class Trainer:
    def __init__(self, hyp, opt, device):
        self.hyp = hyp
        self.opt = opt
        self.device = device
        
        self.set_config()

        self.create_model()

        self.create_optimizer()

        self.load_model_and_values()

        if self.opt.mixed_precision:
            self.init_apex()

        self.create_scheduler()

        self.set_model_mode_and__ema()

        self.create_loaders()

        self.set_model_params()

        if (self.is_world_master()):
            self.create_labels_and_checkanchors()

    def set_config(self):
        def set_directory():
            self.log_dir = self.tb_writer.log_dir if self.tb_writer else 'runs/evolution'  # run directory
            self.wdir = str(Path(self.log_dir) / 'weights') + os.sep  # weights directory
            os.makedirs(self.wdir, exist_ok=True)
            self.last = self.wdir + 'last.pt'
            self.best = self.wdir + 'best.pt'
            self.results_file = self.log_dir + os.sep + 'results.txt'
        
        def create_tb_writer():
                print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
                self.tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', self.opt.name))

        if (self.is_world_master()): create_tb_writer()
        set_directory()

        self.epochs, self.batch_size, self.total_batch_size, self.weights, self.rank, self.mixed_precision = \
                self.opt.epochs, self.opt.batch_size, self.opt.total_batch_size, self.opt.weights, self.opt.local_rank, self.opt.mixed_precision

        

        # Save run settings
        with open(Path(self.log_dir) / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with open(Path(self.log_dir) / 'opt.yaml', 'w') as f:
            yaml.dump(vars(self.opt), f, sort_keys=False)

        # Configure
        init_seeds(2 + self.rank)
        with open(self.opt.data) as f:
            self.data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        self.train_path = self.data_dict['train']
        self.test_path = self.data_dict['val']
        self.nc, self.names = (1, ['item']) if self.opt.single_cls else (int(self.data_dict['nc']), self.data_dict['names'])  # number classes, names
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(self.names), self.nc, self.opt.data)  # check

        # Remove previous results
        if (self.is_world_master()):
            for f in glob.glob('*_batch*.jpg') + glob.glob(self.results_file):
                os.remove(f) 

    def init_apex(self):
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)

    def load_model_and_values(self):
        def load_model(model, opt, weights):
            # load model
            try:
                exclude = ['anchor']  # exclude keys
                ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                                if k in model.state_dict() and not any(x in k for x in exclude)
                                and model.state_dict()[k].shape == v.shape}
                model.load_state_dict(ckpt['model'], strict=False)
                print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), weights))
            except KeyError as e:
                s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                    "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                    % (weights, opt.cfg, weights, weights)
                raise KeyError(s) from e

        def load_optimizer(optimizer):
            # load optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

        def load_results():
            # load results
            if ckpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt

        def load_epoch():
            # epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.epochs < self.start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                    (self.weights, ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']  # finetune additional epochs

        # Load Model
        with torch_distributed_zero_first(self.rank):
            google_utils.attempt_download(self.weights)
        self.start_epoch, self.best_fitness = 0, 0.0
        if self.weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint

            load_model(self.model, self.opt, self.weights)
            load_optimizer(self.optimizer)
            load_results()
            load_epoch()

            del ckpt

    def set_model_mode_and__ema(self):
        # DP mode
        if self.device.type != 'cpu' and self.rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.device.type != 'cpu' and self.rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            print('Using SyncBatchNorm()')

        # Exponential moving average
        self.ema = torch_utils.ModelEMA(self.model) if self.rank in [-1, 0] else None

        # DDP mode
        if self.device.type != 'cpu' and self.rank != -1:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)

    def set_model_params(self):
        # Model parameters
        self.hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(self.device)  # attach class weights
        self.model.names = self.names

    def create_model(self):
        self.model = Model(self.opt.cfg, nc=self.nc).to(self.device)

        # Image sizes
        self.gs = int(max(self.model.stride))  # grid size (max stride)
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in self.opt.img_size]  # verify imgsz are gs-multiples

    def create_optimizer(self):
        # Optimizer
        self.nbs = 64  # nominal batch size
        # default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
        # all-reduce operation is carried out during loss.backward().
        # Thus, there would be redundant all-reduce communications in a accumulation procedure,
        # which means, the result is still right but the training speed gets slower.
        # TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
        # in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
        self.accumulate = max(round(self.nbs / self.total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.total_batch_size * self.accumulate / self.nbs  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else

        if self.hyp['optimizer'] == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2 

    def create_scheduler(self):
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self.lf = lambda x: (((1 + math.cos(x * math.pi / self.epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # plot_lr_scheduler(optimizer, scheduler, epochs)

    def create_loaders(self):
        def create_train_loader():
            # Trainloader
            self.dataloader, self.dataset = create_dataloader(self.train_path, self.imgsz, self.batch_size, self.gs, self.opt, hyp=self.hyp, augment=True,
                                                    cache=self.opt.cache_images, rect=self.opt.rect, local_rank=self.rank,
                                                    world_size=self.opt.world_size)
            mlc = np.concatenate(self.dataset.labels, 0)[:, 0].max()  # max label class
            self.nb = len(self.dataloader)  # number of batches
            assert mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

        def create_test_loader():
            # local_rank is set to -1. Because only the first process is expected to do evaluation.
            self.testloader = create_dataloader(self.test_path, self.imgsz_test, self.total_batch_size, self.gs, self.opt, hyp=self.hyp, augment=False,
                                        cache=self.opt.cache_images, rect=True, local_rank=-1, world_size=self.opt.world_size)[0]

        create_train_loader()
        if (self.is_world_master): create_test_loader()

    def create_labels_and_checkanchors(self):
        # Class frequency
        self.labels = np.concatenate(self.dataset.labels, 0)
        c = torch.tensor(self.labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        plot_labels(self.labels, save_dir=self.log_dir)
        if self.tb_writer:
            # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
            self.tb_writer.add_histogram('classes', c, 0)

        # Check anchors
        if not self.opt.noautoanchor:
            check_anchors(self.dataset, model=self.model, thr=self.hyp['anchor_t'], imgsz=self.imgsz)

    def fit(self):
        def generate_indices(dataset, nc):
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                    k=dataset.n)  # rand weighted idx

        def broadcast_indices(rank, dataset):
            indices = torch.zeros([dataset.n], dtype=torch.int)
            if rank == 0:
                indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
            dist.broadcast(indices, 0)
            if rank != 0:
                dataset.indices = indices.cpu().numpy()

        def set_pbar(mloss):
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, self.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            return s

        def plot():
            f = str(Path(self.log_dir) / ('train_batch%g.jpg' % ni))  # filename
            result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
            if self.tb_writer and result is not None:
                self.tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                # tb_writer.add_graph(model, imgs)  # add model to tensorboard

        def rename_file_and_strip_optimizer():
            # Strip optimizers
            n = ('_' if len(self.opt.name) and not self.opt.name.isnumeric() else '') + self.opt.name
            fresults, flast, fbest = 'results%s.txt' % n, self.wdir + 'last%s.pt' % n, self.wdir + 'best%s.pt' % n
            for f1, f2 in zip([self.wdir + 'last.pt', self.wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    ispt = f2.endswith('.pt')  # is *.pt
                    strip_optimizer(f2) if ispt else None  # strip optimizer
                    os.system('gsutil cp %s gs://%s/weights' % (f2, self.opt.bucket)) if self.opt.bucket and ispt else None  # upload

        # Start training
        t0 = time.time()
        nw = max(3 * self.nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
        maps = np.zeros(self.nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        if self.is_world_master():
            print('Image sizes %g train, %g test' % (self.imgsz, self.imgsz_test))
            print('Using %g dataloader workers' % self.dataloader.num_workers)
            print('Starting training for %g epochs...' % self.epochs)
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.start_epoch, self.epochs):  # epoch ------------------------------------------------------------------
            self.model.train()

            # Update image weights (optional)
            # When in DDP mode, the generated indices will be broadcasted to synchronize dataset.
            if self.dataset.image_weights:
                # Generate indices.
                if (self.is_world_master()):
                    generate_indices(self.dataset, self.nc)
                # Broadcast.
                if self.rank != -1:
                    broadcast_indices(self.rank, self.dataset)

            # Update mosaic border
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(4, device=self.device)  # mean losses
            if self.rank != -1:
                self.dataloader.sampler.set_epoch(epoch)
            pbar = enumerate(self.dataloader)
            if self.is_world_master():
                print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
                pbar = tqdm(pbar, total=self.nb)  # progress bar
            self.optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + self.nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                    accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.total_batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [0.9, self.hyp['momentum']])

                # Multi-scale
                if self.opt.multi_scale:
                    sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                pred = self.model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets.to(self.device), self.model)  # scaled by batch_size
                if self.rank != -1:
                    loss *= self.opt.world_size  # gradient averaged between devices in DDP mode
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Backward
                if self.opt.mixed_precision:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize
                if ni % accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.ema is not None:
                        self.ema.update(self.model)

                # Print
                if (self.is_world_master()):
                    s = set_pbar(mloss)

                    # Plot
                    if ni < 3:
                        plot()

                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            self.scheduler.step()

            # Only the first process in DDP mode is allowed to log or save checkpoints.
            if self.is_world_master():
                # mAP
                if self.ema is not None:
                    self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
                final_epoch = (epoch + 1 == self.epochs)
                if not self.opt.notest or final_epoch:  # Calculate mAP
                    results, maps, times = test.test(self.opt.data,
                                                    batch_size=self.total_batch_size,
                                                    imgsz=self.imgsz_test,
                                                    save_json=final_epoch and self.opt.data.endswith(os.sep + 'coco.yaml'),
                                                    model=self.ema.ema.module if hasattr(self.ema.ema, 'module') else self.ema.ema,
                                                    single_cls=self.opt.single_cls,
                                                    dataloader=self.testloader,
                                                    save_dir=self.log_dir)

                    # Write
                    with open(self.results_file, 'a') as f:
                        f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                    if len(self.opt.name) and opt.bucket:
                        os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, self.opt.bucket, self.opt.name))

                    # Tensorboard
                    if self.tb_writer:
                        tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                                'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                        for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                            self.tb_writer.add_scalar(tag, x, epoch)

                    # Update best mAP
                    fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
                    if fi > self.best_fitness:
                        self.best_fitness = fi

                # Save model
                save = (not self.opt.nosave) or (final_epoch and not self.opt.evolve)
                if save:
                    with open(self.results_file, 'r') as f:  # create checkpoint
                        ckpt = {'epoch': epoch,
                                'best_fitness': self.best_fitness,
                                'training_results': f.read(),
                                'model': self.ema.ema.module if hasattr(self.ema, 'module') else self.ema.ema,
                                'optimizer': None if final_epoch else self.optimizer.state_dict()}

                    # Save last, best and delete
                    torch.save(ckpt, self.last)
                    if (self.best_fitness == fi) and not final_epoch:
                        torch.save(ckpt, best)
                    del ckpt
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training

        if self.is_world_master():
            rename_file_and_strip_optimizer()
            # Finish
            if not self.opt.evolve:
                plot_results(save_dir=self.log_dir)  # save as results.png
            print('%g epochs completed in %.3f hours.\n' % (epoch - self.start_epoch + 1, (time.time() - t0) / 3600))

        return results

    def is_local_master(self) -> bool:
        return self.opt.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        return self.opt.local_rank == -1 or torch.distributed.get_rank() == 0