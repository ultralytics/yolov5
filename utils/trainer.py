import json
import time
from typing import List
import logging
from pathlib import Path

from tqdm import tqdm
import torch
import torch.distributed as dist

from utils.utils import compute_loss
from utils.torch_utils import SequentialDistributedSampler
import utils.torch_utils as torch_utils
from utils.utils import non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh, clip_coords, plot_images, output_to_target


class Trainer:
    def __init__(self, model, rank=-1):
        self.model = model
        self.device = next(model.parameters()).device  # get model device
        self.rank = rank

    def _gather_specific_flexible_tensor(self, tensor_list: List[torch.Tensor], max_dim0_len, fix_dim1_len, dataset_length):
        """
        Args:
            tensor_list: List of tensors which have shape of 2-D. Dim0 length is flexible while Dime1 length is fixed.
        """
        if not tensor_list:
            return []
        t = tensor_list[0]
        full_tensor = torch.empty((len(tensor_list), max_dim0_len, fix_dim1_len+1), device=self.device, dtype=t.dtype)
        for i, tensor in enumerate(tensor_list):
            length = tensor.shape[0]
            full_tensor[i][length:, -1] = 0
            if length:
                full_tensor[i][:length, :-1] = tensor
                full_tensor[i][:length, -1] = 1
        full_tensor = self._distributed_concat(full_tensor, dataset_length)
        results = []
        for tensor in full_tensor:
            results.append(tensor[tensor[:, -1] == 1, :-1])
        return results

    def _distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output

    def is_major_process(self):
        if self.rank in [-1, 0]:
            return True
        else:
            return False

    @staticmethod
    def plot_images_hook(model, batch_i, images, targets, paths, shapes, preds_batch,
            save_dir):
        names = model.names if hasattr(model, 'names') else model.module.names
        if batch_i < 1:
            _, _, height, width = images.shape
            f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            plot_images(images, targets, paths, str(f), names)  # ground truth
            f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
            plot_images(images, output_to_target(preds_batch, width, height), paths, str(f), names)  # predictions

    def infer(self, dataloader, augment, nms_kwargs, training=False, hooks=None):
        if self.rank != -1:
            assert isinstance(dataloader.sampler, SequentialDistributedSampler), type(dataloader.sampler)
        dataset = dataloader.dataset

        # TODO: all reduce these in the end.
        statistics = {
            "t0": torch.zeros(1, device=self.device),
            "t1": torch.zeros(1, device=self.device),
            "loss": torch.zeros(3, device=self.device),
            "batch_size": dataloader.batch_size,
        }
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
        if self.is_major_process():
            dataloader = tqdm(dataloader)
        for batch_i, (images, targets, paths, shapes) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            images = images.half() if half else images.float()  # uint8 to fp16/32
            images /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(self.device)

            batch_size, channels, height, width = images.shape
            whwh = torch.Tensor([width, height, width, height]).to(self.device)

            # Disable gradients
            with torch.no_grad():
                # Run model
                if self.rank != -1:
                    dist.barrier()
                t = torch_utils.time_synchronized()
                inf_out, train_out = model(images, augment=augment)  # inference and training outputs
                statistics["t0"] += torch_utils.time_synchronized() - t

                # Compute loss
                if training:  # if model has loss hyperparameters
                    assert not augment, "otherwise the following code will dump."
                    statistics["loss"] += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

                # Run NMS
                t = torch_utils.time_synchronized()
                preds_batch = non_max_suppression(inf_out, **nms_kwargs)
                statistics["t1"] += torch_utils.time_synchronized() - t

                if hooks is not None:
                    for hook in hooks:
                        hook(model, batch_i, images, targets, paths, shapes, preds_batch)

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

        # All-gather
        if self.rank != -1:
            t0 = time.time()
            preds_list = self._gather_specific_flexible_tensor(preds_list, 300, 6, len(dataset))
            labels_list = self._gather_specific_flexible_tensor(labels_list, 300, 5, len(dataset))
            box_rescaled_list = self._gather_specific_flexible_tensor(box_rescaled_list, 300, 4, len(dataset))
            logging.debug("All gather tensors. Time consumed: %0.2f" % (time.time() - t0))
            dist.all_reduce(statistics["t0"])
            dist.all_reduce(statistics["t1"])
            dist.all_reduce(statistics["loss"])
            statistics["batch_size"] *= dist.get_world_size()

        model.float()  # for training
        return (preds_list, labels_list, box_rescaled_list), statistics
