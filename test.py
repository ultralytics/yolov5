import glob
import os.path as osp

import cv2
import numpy as np
import RRDBNet_arch as arch
import torch
from cv2 import dnn_superres

model_path = '/home/gabriel/projeto/yolov5/modelsResolution/ESPCN_x4.pb'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # if you want to run on GPU, change 'cpu' -> cuda
# device = torch.device('cuda')

test_img_folder = '/home/gabriel/projeto/yolov5/runs/detect/crops/*'

sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel('espcn', 4)

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print(f'Model path {model_path:s}. \nTesting...')

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(f'results/{base:s}_rlt.png', output)
