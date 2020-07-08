# Notes for NVIDIA Xavier NV on docker

## Build a base image, extending dusty_nv's image  
 
    sudo docker build -t yolov5-base -f aarch64.Dockerfile .
 
## Build an image with app specifics, based on the base image
 
    sudo docker build -t yolov5-app -f yolov5.app.Dockerfile .
 
## Use a script to invoke the image for detection with mounted volumes. 
   
Default weights will be pulled the first time into ./weights/
and you should put images into ./inference/. Output will be in
./inference/output/
   
    ./run_xavier_detect.sh

Currently there is a kludge in place to make the following script work. It will 
put sample images into the inference/output/ directory.

#!/bin/bash
# run_from_RTSP_cam.sh

sudo docker run -it --rm \
  --runtime nvidia \
  --network host \
  --mount src=$(pwd)/weights,target=/usr/src/app/weights/,type=bind \
  --mount src=$(pwd)/inference,target=/usr/src/app/inference/,type=bind \
  yolov5-app:latest python3 detect.py --source rtsp://admin:password@192.168.0.31:554//h264Preview_01_sub


