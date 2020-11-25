#!/bin/bash

ROOT_DIR=$(pwd)
export PYTHONPATH=${ROOT_DIR}
echo

YOLO_LEARN_DIR=${ROOT_DIR}/yolo_learn
export PYTHONPATH=${PYTHONPATH}:${YOLO_LEARN_DIR}
echo $PYTHONPATH

