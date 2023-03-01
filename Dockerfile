# Dockerfile
FROM python:3.9.12-slim-buster

RUN mkdir /myapp

COPY . /myapp

WORKDIR /myapp

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        python3-dev \
        python3-setuptools \
        flask \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /myapp/requirement.txt 

    EXPOSE 8080

# CMD ["bash", "run.sh"]

CMD ["python", "main.py"]