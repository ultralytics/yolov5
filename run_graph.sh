#!/bin/bash

cd yolo_learn
python yolo_learn.py --target GRAPH --op_mode TRAIN_TEST --ini_fname yolo_learn_graph.ini
cd ..
