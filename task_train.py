"""
Script to train model.
"""
import logging
import os
import time

import boto3
import botocore
import torch
from bedrock_client.bedrock.api import BedrockApi

from train import trainer

BUCKET_NAME = os.getenv("BUCKET_NAME")
DATA_DIR = os.getenv("DATA_DIR")
EXECUTION_DATE = os.getenv("EXECUTION_DATE")


def list_files(bucket_name, prefix):
    """Get partitions in prefix folder."""
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket_name)
    return [f.key.split(prefix)[1] for f in s3_bucket.objects.filter(Prefix=prefix).all()]


def download_file(key, dest):
    """Download file from S3."""
    s3 = boto3.resource("s3")

    try:
        s3.Bucket(BUCKET_NAME).download_file(key, dest)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


def download_data(date_partitions):
    """Download data from S3 to local by combining all files from the chosen data_partitions.

    Folder structure in S3:
    shellfish
    ├── date_partition=2020-10-01
    │   ├── train
    │   │   ├── images
    │   │   │   ├── img0.jpg
    │   │   │   ├── img1.jpg
    │   │   │   ├── ...
    │   │   │
    │   │   ├── labels
    │   │   │   ├── img0.txt
    │   │   │   ├── img1.txt
    │   │   │   ├── ...
    │   │   │
    │   ├── valid
    │   │   ├── images
    │   │   │   ├── img0.jpg
    │   │   │   ├── img1.jpg
    │   │   │   ├── ...
    │   │   │
    │   │   ├── labels
    │   │   │   ├── img0.txt
    │   │   │   ├── img1.txt
    │   │   │   ├── ...
    │
    ├── date_partition=2020-10-02
    │   ├── ...
    │
    ├── ...

    Folder structure in local:
    img_data
    ├── train
    │   │   ├── images
    │   │   ├── img0.jpg
    │   │   ├── img1.jpg
    │   │   ├── ...
    │   │
    │   ├── labels
    │   │   ├── img0.txt
    │   │   ├── img1.txt
    │   │   ├── ...
    │   │
    ├── valid
    │   ├── images
    │   │   ├── img0.jpg
    │   │   ├── img1.jpg
    │   │   ├── ...
    │   │
    │   ├── labels
    │   │   ├── img0.txt
    │   │   ├── img1.txt
    │   │   ├── ...
    """
    for folder in ["img_data/train/images", "img_data/train/labels",
                   "img_data/valid/images", "img_data/valid/labels"]:
        os.makedirs(folder)

    for mode in ["train", "valid"]:
        for ttype in ["images", "labels"]:
            for date_partition in date_partitions:
                prefix = f"{DATA_DIR}/date_partition={date_partition}/{mode}/{ttype}/"
                files = list_files(BUCKET_NAME, prefix)
                for key in files:
                    download_file(key, f"img_data/{mode}/{ttype}/{key.split(prefix)[1]}")


def compute_log_metrics():
    """Compute and log metrics."""
    # Validation results found in the last 7 elements of the last line of results.txt
    with open("./runs/exp0_yolov5s_results/results.txt", "r") as f:
        lines = f.readlines()
    precision, recall, map50, map50_95, val_giou, val_obj, val_cls = [float(v) for v in lines[-1].split()[-7:]]

    print(f"  Precision          = {precision:.6f}")
    print(f"  Recall             = {recall:.6f}")
    print(f"  mAP@0.5            = {map50:.6f}")
    print(f"  mAP@0.5:0.95       = {map50_95:.6f}")
    print(f"  val GIoU           = {val_giou:.6f}")
    print(f"  val Objectness     = {val_obj:.6f}")
    print(f"  val Classification = {val_cls:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("mAP@0.5", map50)
    bedrock.log_metric("mAP@0.5:0.95", map50_95)
    bedrock.log_metric("val GIoU", val_giou)
    bedrock.log_metric("val Objectness", val_obj)
    bedrock.log_metric("val Classification", val_cls)


def train():
    """Train"""
    print("PyTorch Version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device found = {device}")

    if device.type == "cuda":
        print("  Number of GPUs:", torch.cuda.device_count())
        print("  Device properties:", torch.cuda.get_device_properties(0))

    print("\nDownload data")
    start = time.time()
    download_data(date_partitions=[EXECUTION_DATE])
    print(f"  Time taken = {time.time() - start:.0f} secs")

    print("\nTrain model")
    params = {
        'weights': '',
        'cfg': './models/custom_yolov5s.yaml',
        'data': 'data.yaml',
        'epochs': 2,
        'batch_size': 16,
        'img_size': [416],
        'cache_images': True,
        'name': 'yolov5s_results',
    }
    trainer(params)

    print("\nEvaluate")
    compute_log_metrics()


if __name__ == "__main__":
    train()
