# This file contains google utils: https://cloud.google.com/storage/docs/reference/libraries
# pip install --upgrade google-cloud-storage
# from google.cloud import storage

import os
import time
from pathlib import Path


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    msg = weights + ' missing, try downloading from https://drive.google.com/drive/folders/1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J'

    r = 1  # return
    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'yolov3-spp.pt': '1mM67oNw4fZoIOL1c8M3hHmj66d8e-ni_',  # yolov3-spp.yaml
             'yolov5s.pt': '1R5T6rIyy3lLwgFXNms8whc-387H0tMQO',  # yolov5s.yaml
             'yolov5m.pt': '1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr',  # yolov5m.yaml
             'yolov5l.pt': '1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV',  # yolov5l.yaml
             'yolov5x.pt': '1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS',  # yolov5x.yaml
             }

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)

        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
            s = "curl -L -o %s 'storage.googleapis.com/ultralytics/yolov5/ckpt/%s'" % (weights, file)
            r = os.system(s)  # execute, capture return values

            # Error check
            if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                raise Exception(msg)


def gdrive_download(id='1n_oKgR81BJtqk75b00eAjdv03qVCQn2f', name='coco128.zip'):
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    os.remove(name) if os.path.exists(name) else None  # remove existing
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    os.system("curl -c ./cookie -s -L \"drive.google.com/uc?export=download&id=%s\" > /dev/null" % id)
    if os.path.exists('cookie'):  # large file
        s = "curl -Lb ./cookie \"drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=%s\" -o %s" % (
            id, name)
    else:  # small file
        s = "curl -s -L -o %s 'drive.google.com/uc?export=download&id=%s'" % (name, id)
    r = os.system(s)  # execute, capture return values
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    if r != 0:
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))
    return r


# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
