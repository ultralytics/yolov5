import os
import cv2
import torch
from PIL import Image

import supervisely_lib as sly

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

remote_weights_path = "/yolov5_train/coco128_002/2278_071/weights/last.pt"
local_weights_path = None

model = None


def download_weights():
    global local_weights_path
    local_weights_path = remote_weights_path
    sly.fs.ensure_base_path(local_weights_path)
    my_app.public_api.file.download(TEAM_ID, remote_weights_path, local_weights_path)


def deploy_model():
    for f in ['zidane.jpg', 'bus.jpg']:  # download 2 images
        print(f'Downloading {f}...')
        torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/' + f, f)

    global model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=local_weights_path)
    x = 10

    img1 = Image.open('zidane.jpg')  # PIL image
    img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)

    results = model(img1)

    # Results
    results.print()  # print results to screen
    results.show()  # display results
    results.save()  # save as results1.jpg, results2.jpg... etc.

    x = 10
    x += 1
    names = model.module.names if hasattr(model, 'module') else model.names


def main():
    download_weights()
    deploy_model()
    pass


if __name__ == "__main__":
    sly.main_wrapper("main", main)