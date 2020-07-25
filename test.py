import argparse
import json
import functools
from typing import List
import logging

import torch
import torch.distributed as dist

from models.experimental import *
from utils.datasets import *
from utils.utils import compute_loss
from utils.torch_utils import SequentialDistributedSampler, torch_distributed_zero_first
from utils.trainer import Trainer
from utils.evaluation import do_evaluation


def test(data,
         device,
         opt,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=True,
         save_dir='',
         merge=False,
         save_txt=False,
         local_rank=-1
    ):
    # Initialize/load model and set device
    device = torch_utils.select_device(device, batch_size=batch_size)
    # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if device.type != 'cpu' and local_rank != -1:
        # DDP mode
        assert torch.cuda.device_count() > local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

        world_size = dist.get_world_size()
        assert batch_size % world_size == 0, "Batch size is not a multiple of the number of devices given!"
        batch_size = batch_size // world_size

    #TODO: save_txt implementation.
    #if save_txt:
    #    out = Path('inference/output')
    #    if os.path.exists(out):
    #        shutil.rmtree(out)  # delete output folder
    #    os.makedirs(out)  # make new output folder

    # Remove previous
    if local_rank in [-1, 0]:
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)

    # Load model
    with torch_distributed_zero_first(local_rank):
        model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images

    # Dataloader
    sampler = functools.partial(SequentialDistributedSampler, batch_size=batch_size) if local_rank != -1 else None
    dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                    hyp=None, augment=False, cache=False, pad=0.5, rect=True, sampler=sampler)[0]

    if device.type != 'cpu' and local_rank == -1 and torch.cuda.device_count() > 1:
        # TODO: Still doesn't work.
        raise NotImplementedError
        model = torch.nn.DataParallel(model)

    # run once
    if device.type != 'cpu':
        if half:
            model.half()
        img = torch.zeros((torch.cuda.device_count()*2, 3, imgsz, imgsz), device=device)  # init img
        with torch.no_grad():
            model(img.half() if half else img)

    nms_kwargs = {"conf_thres":conf_thres, "iou_thres":iou_thres, "merge":merge}

    hooks = []
    if not save_dir:
        hooks.append(functools.partial(Trainer.plot_images_hook, save_dir=save_dir))
    trainer = Trainer(model, rank=local_rank)
    infer_results, infer_statistics = trainer.infer(dataloader, augment, nms_kwargs, training=False, hooks=hooks)

    fname = 'detections_val2017_%s_results.json' % \
        (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename

    if local_rank in [-1, 0]:
        return do_evaluation(infer_results, infer_statistics, model, dataloader.dataset, nc,
            verbose=verbose,
            do_official_coco_evaluation=save_json,
            official_coco_evaluation_save_fname=fname)
    else:
        return (None, None, None, None, None), None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--local_rank', type=int, help='Reserved for DDP.', default=-1)
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.device,
             opt,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             merge=opt.merge,
             save_txt=opt.save_txt,
             local_rank=opt.local_rank)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(352, 832, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, opt.device, opt,
                    weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                    merge=opt.merge,
                    save_txt=opt.save_txt)
                y.append(r + t)  # results and times
            if opt.local_rank in [-1, 0]:
                np.savetxt(f, y, fmt='%10.4g')  # save
        if opt.local_rank in [-1, 0]:
            os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # plot
