import argparse
import os
import torch
import time
import numpy as np
from models.experimental import attempt_load

NUM_IMAGES = 3
TENSOR_TYPE = torch.half

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
model = attempt_load(pt_file, map_location=device).to(TENSOR_TYPE)
model.eval()

proc_time = []
with torch.no_grad():
    dummy_input = torch.zeros(1,3,640,640).to(TENSOR_TYPE).to(device)
    for i in range(20):
        t1 = time.time()
        for _ in range(NUM_IMAGES):
            pred = model(dummy_input)
        diff = time.time() - t1
        proc_time.append(diff*1000)
        print(f'proc time: {diff:.4f}')
proc_time = np.array(proc_time[1:])
print(f'mean proc time: {proc_time.mean():.2f} ms')
print(f'max proc time: {proc_time.max():.2f} ms')
print(f'min proc time: {proc_time.min():.2f} ms')
