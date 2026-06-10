# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Builds ultralytics/yolov5:latest image on DockerHub https://hub.docker.com/r/ultralytics/yolov5
# Image is CUDA-optimized for YOLOv5 single/multi-GPU training and inference

# Start FROM PyTorch image https://hub.docker.com/r/pytorch/pytorch
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl

# Create working directory
RUN rm -rf /usr/src/app && mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Install pip packages
COPY requirements.txt .
RUN python -m pip install uv wheel
RUN uv pip install --prefix "$(python -c 'import sys; print(sys.prefix)')" --no-cache -r requirements.txt albumentations comet_ml gsutil notebook \
    coremltools onnx onnxslim onnxruntime 'openvino>=2024.0.0'

# Set environment variables
ENV OMP_NUM_THREADS=1

# Cleanup
ENV DEBIAN_FRONTEND=teletype


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -f utils/docker/Dockerfile -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# Clean up
# sudo docker system prune -a --volumes

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/ultralytics/yolov5:latest
