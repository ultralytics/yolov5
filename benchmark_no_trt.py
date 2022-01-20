import argparse
import os
import torch
import time
import numpy as np
from models.experimental import attempt_load

NUM_IMAGES = 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    return args.weights


pt_file = parse_args()

if torch.cuda.is_available():
    print('using GPU')
    device = torch.device('cuda:0')
else:
    print('using CPU')
    device = torch.device('cpu')

# Load model
model = attempt_load(pt_file, map_location=device).float()
model.eval()

with torch.no_grad():
    dummy_input = torch.ones(1,3,640,640)*0.1
    dummy_input = dummy_input.cuda()
    proc_time = []
    for i in range(20):
        t1 = time.time()
        for j in range(NUM_IMAGES):
            pred = model(dummy_input)
        diff = time.time() - t1
        proc_time.append(diff*1000)
        print('proc time: {:.4f}'.format(diff*1000))
    proc_time = np.array(proc_time[1:])
    print('mean proc time: {:.4f}'.format(proc_time.mean()))
    print('max proc time: {:.4f}'.format(proc_time.max()))
    print('min proc time: {:.4f}'.format(proc_time.min()))
