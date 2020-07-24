import argparse
import json
import functools
from typing import List

import torch
import torch.distributed as dist

from models.experimental import *
from utils.datasets import *
from utils.metrics import MetricMAP, MetricCoco
from utils.utils import compute_loss
from utils.torch_utils import SequentialDistributedSampler


class Inferencer:
    def __init__(self, model, local_rank=-1):
        self.model = model
        self.device = next(model.parameters()).device  # get model device
        self.local_rank = local_rank
        self.init_statistics()

    def init_statistics(self):
        self.t0, self.t1 = 0, 0
        self.loss = torch.zeros(3, device=self.device)

    def _gather_specific_flexible_tensor(self, tensor_list: List[torch.Tensor], max_dim1_len, dataset_length):
        """
        Args:
            tensor_list: List of tensors which have shape of 2-D. Dim1 length is flexible while Dime2 length is fixed.
        """
        t = tensor_list[0]
        full_tensor = torch.empty((len(tensor_list), max_dim1_len, t.shape[1]+1), device=self.device, dtype=t.dtype)
        for i, tensor in enumerate(tensor_list):
            length = tensor.shape[0]
            full_tensor[i][length:, -1] = 0
            if length:
                full_tensor[i][:length, :-1] = tensor
                full_tensor[i][:length, -1] = 1
        full_tensor = self.distributed_concat(full_tensor, dataset_length)
        results = []
        for tensor in full_tensor:
            results.append(tensor[tensor[:, -1] == 1, :-1])
        return results

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output

    def infer(self, dataloader, augment, nms_kwargs, training=False, hooks=[]):
        if self.local_rank != -1:
            assert isinstance(dataloader.sampler, SequentialDistributedSampler), type(dataloader.sampler)

        model = self.model

        # Half
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Configure
        model.eval()

        # Inference Loop
        preds_list = []
        labels_list = []
        box_rescaled_list = []
        for batch_i, (images, targets, paths, shapes) in enumerate(tqdm(dataloader)):
            images = images.to(self.device, non_blocking=True)
            images = images.half() if half else images.float()  # uint8 to fp16/32
            images /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(self.device)

            batch_size, channels, height, width = images.shape
            whwh = torch.Tensor([width, height, width, height]).to(self.device)

            # Disable gradients
            with torch.no_grad():
                # Run model
                t = torch_utils.time_synchronized()
                inf_out, train_out = model(images, augment=augment)  # inference and training outputs
                self.t0 += torch_utils.time_synchronized() - t

                # Compute loss
                if training:  # if model has loss hyperparameters
                    assert not augment, "otherwise the following code will dump."
                    self.loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

                # Run NMS
                t = torch_utils.time_synchronized()
                preds_batch = non_max_suppression(inf_out, **nms_kwargs)
                self.t1 += torch_utils.time_synchronized() - t

                for hook in hooks:
                    hook(batch_i, images, targets, paths, shapes, preds_batch)

                # Post processing
                for si in range(batch_size):
                    labels = targets[targets[:, 0] == si, 1:]
                    labels[:, 1:5] = xywh2xyxy(labels[:, 1:5]) * whwh
                    labels_list.append(labels)

                    preds = preds_batch[si]
                    # Clip boxes to image bounds
                    if preds is None:
                        preds = torch.zeros(0).to(self.device)
                    if len(preds):
                        clip_coords(preds, (height, width))
                    preds_list.append(preds)

                    if len(preds):
                        box = preds[:, :4].clone()  # xyxy
                        scale_coords((height, width), box, shapes[si][0], shapes[si][1])  # to original shape
                        box = xyxy2xywh(box)  # xywh
                        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    else:
                        box = torch.zeros(0).to(self.device)
                    box_rescaled_list.append(box)

        if self.local_rank != -1:
            t0 = time.time()
            preds_list = self._gather_specific_flexible_tensor(preds_list, 300, len(dataloader.dataset))
            labels_list = self._gather_specific_flexible_tensor(labels_list, 300, len(dataloader.dataset))
            box_rescaled_list = self._gather_specific_flexible_tensor(box_rescaled_list , 300, len(dataloader.dataset))
            print("All gather tensors. Time consumed: %0.2f" % (time.time() - t0))

        model.float()  # for training
        return preds_list, labels_list, box_rescaled_list


def inference_and_eval(model,
        dataloader,
        nms_kwargs,
        nc,
        augment,
        training,
        local_rank = -1,
        save_dir_for_debug_images = None,
        verbose = False,
        do_official_coco_evaluation = False,
        official_coco_evaluation_save_fname = None
    ):
    names = model.names if hasattr(model, 'names') else model.module.names
    device = next(model.parameters()).device  # get model device
    imgsz = dataloader.dataset.img_size

    ##############################
    # Model Inference
    # Plot images
    hooks = []
    if not save_dir_for_debug_images:
        def plot_images_hook(batch_i, images, targets, paths, shapes, preds_batch):
            if batch_i < 1:
                _, _, height, width = images.shape
                f = Path(save_dir_for_debug_images) / ('test_batch%g_gt.jpg' % batch_i)  # filename
                plot_images(images, targets, paths, str(f), names)  # ground truth
                f = Path(save_dir_for_debug_images) / ('test_batch%g_pred.jpg' % batch_i)
                plot_images(images, output_to_target(preds_batch, width, height), paths, str(f), names)  # predictions
        hooks.append(plot_images_hook)

    inferencer = Inferencer(model, local_rank=local_rank)
    inference_results = inferencer.infer(dataloader, augment, nms_kwargs, training, hooks=hooks)
    # TODO: all reduce these.
    t0 = inferencer.t0
    t1 = inferencer.t1
    loss = inferencer.loss

    if local_rank in [0, -1]:
        ##############################
        # Calculate speeds
        t = tuple(x / len(dataloader.dataset) * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, dataloader.batch_size)  # tuple
        # Speed is presently only valid in single-gpu mode.
        if not training and local_rank == -1:
            print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

        ##############################
        # Calculate home-made mAP
        iou_vec = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        metric = MetricMAP(iou_vec, nc)
        eval_result = metric.eval(inference_results[0], inference_results[1])
        metric.print_result(eval_result, names, verbose)
        mAP = eval_result.mAP
        mAP50 = eval_result.mAP50

        ##############################
        # Save JSON, Calculate coco mAP
        if do_official_coco_evaluation:
            metric_coco = MetricCoco(dataloader.dataset.img_files)
            mAP_coco_official, mAP50_coco_official = metric_coco.eval(inference_results[0], inference_results[2], official_coco_evaluation_save_fname)
            if mAP_coco_official:
                mAP = mAP_coco_official
            if mAP50_coco_official:
                mAP50 = mAP50_coco_official

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

        if local_rank == 0:
            dist.barrier()
        return (eval_result.mp, eval_result.mr, mAP50, mAP, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    else:
        dist.barrier()
        return (None, None, None, None, None), None, None


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
         verbose=False,
         save_dir='',
         merge=False,
         save_txt=False,
         local_rank=-1
    ):
    # Initialize/load model and set device
    device = torch_utils.select_device(device, batch_size=batch_size)
    if device.type == 'cpu':
        #mixed_precision = False
        pass
    elif local_rank != -1:
        # DDP mode
        assert torch.cuda.device_count() > local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

        world_size = dist.get_world_size()
        assert batch_size % world_size == 0, "Batch size is not a multiple of the number of devices given!"
        batch_size = batch_size // world_size

    if save_txt:
        out = Path('inference/output')
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Remove previous
    for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
        os.remove(f)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # Dataloader
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    if half:
        model.half()
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images

    sampler = None
    if local_rank != -1:
        sampler = functools.partial(SequentialDistributedSampler, batch_size=batch_size)
    dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                    hyp=None, augment=False, cache=False, pad=0.5, rect=True, sampler=sampler)[0]

    nms_kwargs = {"conf_thres":conf_thres, "iou_thres":iou_thres, "merge":merge}
    fname = 'detections_val2017_%s_results.json' % \
        (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename

    return inference_and_eval(model, dataloader, nms_kwargs, nc, augment,
        training=False,
        local_rank=local_rank,
        save_dir_for_debug_images=save_dir,
        verbose=verbose,
        do_official_coco_evaluation=save_json,
        official_coco_evaluation_save_fname=fname)


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
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
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
             opt.verbose,
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
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # plot
