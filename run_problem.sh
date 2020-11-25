#!/bin/bash

cd yolo_learn
python yolo_learn.py --target PROBLEM --op_mode TRAIN_TEST --ini_fname yolo_learn_problem.ini
cd ..
