# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.06-py3
RUN pip install -U gsutil

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Install dependencies (pip or conda)
#RUN pip install -r requirements.txt

# Copy weights
#RUN python3 -c "from models import *; \
#attempt_download('weights/yolov5s.pt'); \
#attempt_download('weights/yolov5m.pt'); \
#attempt_download('weights/yolov5l.pt')"


# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host $t bash

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t bash

# Kill all
# sudo docker kill "$(sudo docker ps -q)"

# Kill all image-based
# sudo docker kill $(sudo docker ps -a -q --filter ancestor=ultralytics/yolov5:latest)

# Run bash for loop
# sudo docker run --gpus all --ipc=host ultralytics/yolov5:latest while true; do python3 train.py --evolve; done

# Bash into running container
# sudo docker container exec -it ba65811811ab bash
# python -c "from utils.utils import *; create_pretrained('weights/last.pt')" && gsutil cp weights/pretrained.pt gs://*

# Bash into stopped container
# sudo docker commit 6d525e299258 user/test_image && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco --entrypoint=sh user/test_image

# Clean up
# docker system prune -a --volumes