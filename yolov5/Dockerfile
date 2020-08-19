# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.03-py3

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt gsutil

# ImportError: libGL.so.1: cannot open shared object file: No such file or directory solution
RUN apt update && apt install -y libgl1-mesa-glx

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Copy weights
#RUN python3 -c "from models import *; \
#attempt_download('weights/yolov5s.pt'); \
#attempt_download('weights/yolov5m.pt'); \
#attempt_download('weights/yolov5l.pt')"


# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t
# for v in {300..303}; do t=ultralytics/coco:v$v && sudo docker build -t $t . && sudo docker push $t; done

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t

# Kill all
# sudo docker kill "$(sudo docker ps -q)"

# Kill all image-based
# sudo docker kill $(sudo docker ps -a -q --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker container exec -it ba65811811ab bash

# Bash into stopped container
# sudo docker commit 092b16b25c5b usr/resume && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco --entrypoint=sh usr/resume

# Send weights to GCP
# python -c "from utils.general import *; strip_optimizer('runs/exp0_*/weights/best.pt', 'tmp.pt')" && gsutil cp tmp.pt gs://*.pt

# Clean up
# docker system prune -a --volumes
