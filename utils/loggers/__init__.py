# YOLOv5 experiment logging utils

import warnings

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import colorstr
from utils.torch_utils import de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger

LOGGERS = ('txt', 'tb', 'wandb')  # text-file, TensorBoard, Weights & Biases

try:
    import wandb
except ImportError:
    wandb = None


class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, results_file=None, weights=None, opt=None, hyp=None,
                 data_dict=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.results_file = results_file
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.data_dict = data_dict
        self.logger = logger  # for printing results to console
        self.include = include
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary

    def start(self):
        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            prefix = colorstr('tensorboard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        try:
            assert 'wandb' in self.include and wandb
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, s.stem, run_id, self.data_dict)
        except:
            self.include = (x for x in self.include if x != 'wandb')
            self.wandb = None

        return self

    def on_train_batch_end(self, ni, model, imgs):
        # Callback runs on train batch end
        if ni == 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress jit trace warning
                self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
        if 'wandb' in self.include and ni == 10:
            files = self.save_dir.glob('train*.jpg')
            wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if 'wandb' in self.include:
            self.wandb.current_epoch = epoch + 1

    def on_val_batch_end(self, pred, predn, path, names, im):
        # Callback runs on train batch end
        if 'wandb' in self.include and self.wandb.wandb_run:
            self.wandb.val_one_image(pred, predn, path, names, im)

    def on_val_end(self):
        # Callback runs on val end
        if 'wandb' in self.include:
            files = sorted(self.save_dir.glob('val*.jpg'))
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})

    def on_train_val_end(self, mloss, results, lr, epoch, s, best_fitness, fi):
        # Callback runs on validation end during training
        vals = list(mloss[:-1]) + list(results) + lr
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'x/lr0', 'x/lr1', 'x/lr2']  # params
        if 'txt' in self.include:
            with open(self.results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
        if 'tb' in self.include:
            for x, tag in zip(vals, tags):
                self.tb.add_scalar(tag, x, epoch)  # TensorBoard
        if 'wandb' in self.include:
            self.wandb.end_epoch(best_result=best_fitness == fi)
            for x, tag in zip(vals, tags):
                wandb.log({tag: x})

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if 'wandb' in self.include:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best):
        # Callback runs on training end
        files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter
        if 'wandb' in self.include:
            wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            wandb.log_artifact(str(best if best.exists() else last), type='model',
                               name='run_' + self.wandb.wandb_run.id + '_model',
                               aliases=['latest', 'best', 'stripped'])
            self.wandb.finish_run()

    def log_images(self, paths):
        # Log images
        if 'wandb' in self.include:
            wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]}, commit=False)
