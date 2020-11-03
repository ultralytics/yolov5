#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-12:00:00
#SBATCH -o slurm.%N.%j.out # STDOUT 
#SBATCH -e slurm.%N.%j.err # STDERR

# Load required modules
module purge
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


# Launch multiple process python code
echo "Searching for deepfakes"
time srun -n 8 python3 train.py --cfg custom_train/yolov1-spp.cfg --data custom_train/train.data --nosave --weights weights/yolov3-spp-ultralytics.pt