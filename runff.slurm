#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH -p gpgpu
#SBATCH -q gpgpumse

module purge
module load fosscuda/2019b
module load cuda/10.1.243
module load gcccore/8.3.0 
module load gcc/8.3.0 openmpi/3.1.4
module load python/3.7.4 
module load opencv
module load pillow
module load torch/20200428
module load scipy-bundle
module load pyyaml
module load numpy/1.17.3-python-3.7.4
module load torchvision
module load matplotlib/3.1.1-python-3.7.4
module load scikit-learn
module load torchvision/0.5.0-python-3.7.4
module load tqdm
module unload pytorch/1.4.0-python-3.7.4 
module load pytorch-geometric/1.6.1-python-3.7.4-pytorch-1.6.0
module load tensorflow/2.3.1-python-3.7.4

time python3 -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 32 --data custom_train/dataset_ff.yaml --weights weights/yolov5l.pt