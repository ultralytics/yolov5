"""
Script for serving.
"""
import os

import torch
from torch.nn import functional as F
from torchvision import transforms
from flask import Flask, request
from captum.attr import GuidedGradCam, IntegratedGradients

from gradcam_pytorch import GradCam
from utils import Rescale, RandomCrop, ToTensor, CustomSEResNeXt, seed_torch
from utils_image import (
    encode_image, decode_image, superimpose_heatmap, get_heatmap, normalize_image_attr)

MODEL_DIR = "/artefact/"
if os.path.exists("models/"):
    MODEL_DIR = "models/"

DEVICE = torch.device("cpu")
MODEL = CustomSEResNeXt(MODEL_DIR + "pretrained_model.pth", DEVICE)
MODEL.load_state_dict(torch.load(MODEL_DIR + "finetuned_model.pth", map_location=DEVICE))
MODEL.eval()

GRAD_CAM = GradCam(model=MODEL.model,
                   feature_module=MODEL.model.layer4,
                   target_layer_names=["2"],
                   use_cuda=False)

GUIDED_GC = GuidedGradCam(MODEL.model, MODEL.model.layer4)
IG = IntegratedGradients(MODEL.model)


def process_image(image):
    """Process image."""
    seed_torch(seed=42)
    proc_image = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor(),
    ])(image)
    proc_image = proc_image.unsqueeze(0).to(DEVICE)
    return proc_image


def predict(proc_image):
    """Predict function."""
    with torch.no_grad():
        logits = MODEL(proc_image)
        prob = F.softmax(logits, dim=1).cpu().numpy()[0, 1].item()
    return prob


def cv_xai(proc_image, prob):
    """Perform XAI."""
    target = 1 if prob > 0.5 else 0

    # Grad-CAM
    img = proc_image.detach().numpy().squeeze().transpose((1, 2, 0))
    mask = GRAD_CAM(proc_image)
    cam_img = superimpose_heatmap(img, mask)

    # Guided Grad-CAM
    gc_attribution = GUIDED_GC.attribute(proc_image, target=target)
    gc_norm_attr = normalize_image_attr(
        gc_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
        sign="absolute_value", outlier_perc=2
    )
    gc_img = get_heatmap(gc_norm_attr)

    # IntegratedGradients
    ig_attribution = IG.attribute(proc_image, target=target, n_steps=20)
    ig_norm_attr = normalize_image_attr(
        ig_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
        sign="absolute_value", outlier_perc=2
    )
    ig_img = get_heatmap(ig_norm_attr)

    return {
        "cam_image": encode_image(cam_img),
        "gc_image": encode_image(gc_img),
        "ig_image": encode_image(ig_img),
    }


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    image = decode_image(request.json["encoded_image"])
    proc_image = process_image(image)
    prob = predict(proc_image)
    output = {"prob": prob}

    xai_imgs = cv_xai(proc_image, prob)
    output.update(xai_imgs)
    return output


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
