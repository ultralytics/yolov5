# Notes for NVIDIA Xavier NV on docker

## Build a base image, extending dusty_nv's image  
 
    sudo docker build -t yolov5-base -f aarch64.Dockerfile .
 
## Build an image with app specifics, based on the base image
 
    sudo docker build -t yolov5-app -f yolov5.app.Dockerfile .
 
## Use a script to invoke the image for detection with mounted volumes. 
   
Default weights will be pulled the first time into ./weights/
and you should put images into ./inference/. Output will be in
./inference/output/
   
    ./run_xavier.sh
