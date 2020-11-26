# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM supervisely/base-py-cuda-10-1

# Install dependencies
RUN pip install --upgrade pip
RUN pip install gsutil


RUN pip install Cython
RUN pip install matplotlib>=3.2.2
RUN pip install numpy>=1.18.5
RUN pip install opencv-python>=4.1.2
RUN pip install pillow
RUN pip install PyYAML>=5.3
RUN pip install scipy>=1.4.1
RUN pip install tensorboard>=2.2
RUN pip install torch>=1.6.0
RUN pip install torchvision>=0.7.0
RUN pip install tqdm>=4.41.0


RUN pip install simplejson
RUN conda install -c conda-forge python-json-logger
RUN pip install prettytable
RUN pip install shapely
RUN pip install pascal_voc_writer
RUN pip install requests_toolbelt
RUN pip install bidict
RUN pip install sk-video


WORKDIR /alex_work
# Create working directory
#RUN mkdir -p /usr/src/app
#WORKDIR /usr/src/app

# Copy contents
#COPY . /usr/src/app

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
# sudo docker kill $(sudo docker ps -q)

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
