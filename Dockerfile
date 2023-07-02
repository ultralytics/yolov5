FROM ubuntu:22.04

ENV NODE_VERSION=16.16.0
RUN apt-get update && apt-get install -y curl

# ensure all directories exist
WORKDIR /app

EXPOSE 8700

COPY requirements.txt /app/requirements.txt

RUN apt-get update
RUN apt-get install -y python3 pip
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN cd /app && pip install -r requirements.txt

CMD ["python3", "/app/server.py"]
