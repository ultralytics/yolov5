import torchvision
import torch
import torch.nn.functional as F

from pycocotools.coco import COCO

import numpy as np

from .loss import ComputeLoss

from hannah.datasets.Kitti import KittiCOCO

class UltralyticsYolo(torch.nn.Module):
    def __init__(
        self,
        name="yolov5s",
        num_classes=80,
        pretrained=True,
        autoshape=True,
        force_reload=False,
        gr=1,
        hyp=dict(),
        *args,
        **kwargs,
    ):

        super().__init__()

        # Model
        self.model = torch.hub.load(
            "ultralytics/yolov5" if name.startswith("yolov5") else "ultralytics/yolov3",
            name,
            classes=num_classes,
            pretrained=pretrained,
            autoshape=autoshape,
            force_reload=force_reload,
        )
        self.model.hyp = hyp
        self.model.gr = gr
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def _transformAnns(self, x, y):
        retval = []

        for x_elem, y_elem in zip(x, y):
            img_ann = []
            img_widh = x_elem.shape[2]
            img_height = x_elem.shape[1]

            for box, label in zip(
                (boxes for boxes in y_elem["boxes"]),
                (labels for labels in y_elem["labels"]),
            ):
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                img_ann.append(
                    torch.tensor(
                        [
                            0,
                            label,
                            (box[0] + (box_width / 2)) / img_widh,
                            (box[1] + (box_height / 2)) / img_height,
                            box_width / img_widh,
                            box_height / img_height,
                        ]
                    )
                )
            retval.append(torch.stack(img_ann))
        return retval

    def forward(self, x, y=None):
        if isinstance(x, (tuple, list)):
            retval = list()
            for x_elem in x:
                pad = (
                    (0, 1248 - x_elem.size()[2], 0, 384 - x_elem.size()[1])
                    if "6" not in self.model.yaml_file
                    else (0, 1280 - x_elem.size()[2], 0, 1280 - x_elem.size()[1])
                )
                retval.append(self.model(F.pad(x_elem.unsqueeze(0), pad, "constant")))

            if self.training:
                ret_loss = []

                loss = ComputeLoss(self.model)
                y = self._transformAnns(x, y)

                for x_elem, y_elem in zip(retval, y):
                    ret_loss.append(loss(x_elem, y_elem.to(x_elem[0].device))[0])
                retval = dict(
                    zip((i for i in range(len(ret_loss))), (ret for ret in ret_loss))
                )

            return retval
        else:
            pad = (
                (0, 1248 - x.size()[3], 0, 384 - x.size()[2])
                if "6" not in self.model.yaml_file
                else (0, 1280 - x.size()[3], 0, 1280 - x.size()[2])
            )
            x = F.pad(x, pad, "constant")
            return self.model(x)

    def train(self, mode=True):
        super().train(mode)
        self.model.nms(not mode)

    def transformOutput(self, cocoGt, output, x, y):
        retval = []

        for out, x_elem, y_img in zip(output, x, y):
            for ann in out[0].data:
                x1 = ann[0].item()
                y1 = ann[1].item()
                x2 = ann[2].item()
                y2 = ann[3].item()
                confidence = ann[4].item()
                label = ann[5].item()
                if not KittiCOCO.dontCareMatch(
                    torch.Tensor(np.array([x1, y1, x2, y2])),
                    (x_elem.shape[2], x_elem.shape[1]),
                    y_img,
                ):
                    img_dict = dict()
                    img_dict["image_id"] = cocoGt.getImgId(y_img["filename"])
                    img_dict["category_id"] = label
                    img_dict["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                    img_dict["score"] = confidence
                    retval.append(img_dict)

        if len(retval) == 0:
            return COCO()
        return cocoGt.loadRes(retval)
