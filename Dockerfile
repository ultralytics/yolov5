# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext
RUN pip install --no-cache -r requirements.txt albumentations wandb gsutil notebook \
    torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# RUN pip install --no-cache -U torch torchvision

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

# Set environment variables
# ENV HOME=/usr/src/app


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# Clean up
# docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/ultralytics/yolov5:latest
