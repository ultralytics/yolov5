#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account hpcadmingpgpu
# Use a project ID that has gpgpu access.
#SBATCH --partition shortgpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 00:05:00
#SBATCH --cpus-per-task=1

module purge
source /usr/local/module/spartan_old.sh
module load Tensorflow/1.8.0-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.5.2-GPU

python tensor_flow.py
