# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.03-py3

# Install dependencies (pip or conda)
RUN pip install -U gsutil thop
# RUN pip install -U -r requirements.txt

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

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host $t bash

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t bash

# Kill all
# sudo docker kill "$(sudo docker ps -q)"

# Kill all image-based
# sudo docker kill $(sudo docker ps -a -q --filter ancestor=ultralytics/yolov5:latest)

# Run bash for loop
# sudo docker run --gpus all --ipc=host ultralytics/yolov5:latest while true; do python3 train.py --evolve; done

# Bash in running container
# sudo docker container exec -it 97919ad657de /bin/bash

# Bash last stopped container
# python -c "from utils.utils import *; create_backbone('weights/best.pt')" && gsutil cp weights/backbone.pt gs://ult/coco/yolov5s.pt

# Clean up
# docker system prune -a --volumes