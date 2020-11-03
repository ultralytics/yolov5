#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgputest
#SBATCH --gres=gpu:p100:1
#SBATCH --time 00:05:00
#SBATCH --cpus-per-task=1

module purge
module load fosscuda/2019b
module load tensorflow/2.1.0-python-3.7.4

python3 tensor_flow3.py
