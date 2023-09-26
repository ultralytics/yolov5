FROM ubuntu:22.04

ENV NODE_VERSION=16.16.0
RUN apt-get update && apt-get install -y curl python3 pip ffmpeg libsm6 libxext6

# ensure all directories exist
WORKDIR /app
USER root

COPY requirements.txt /app/requirements.txt
RUN cd /app && pip install -r requirements.txt

EXPOSE 8700
# COPY . /app/

# CMD ["python3", "/app/server.py"]
