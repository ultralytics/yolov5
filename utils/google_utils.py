# Google utils: https://cloud.google.com/storage/docs/reference/libraries

import os
import platform
import subprocess
import time
from pathlib import Path

import requests
import torch


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output('gsutil du %s' % url, shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = str(weights).strip().replace("'", '')
    file = Path(weights).name.lower()

    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    response = requests.get('https://api.github.com/repos/ultralytics/yolov5/releases/latest').json()  # github api
    assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
    redundant = False  # second download option

    if file in assets and not os.path.isfile(weights):
        try:  # GitHub
            tag = response['tag_name']  # i.e. 'v1.0'
            url = f'https://github.com/ultralytics/yolov5/releases/download/{tag}/{file}'
            print('Downloading %s to %s...' % (url, weights))
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            print('Download error: %s' % e)
            assert redundant, 'No secondary mirror'
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            print('Downloading %s to %s...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))  # torch.hub.download_url_to_file(url, weights)
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # check
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                print('ERROR: Download failure: %s' % msg)
            print('')
            return


def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', name='tmp.zip'):
    # Downloads a file from Google Drive. from yolov5.utils.google_utils import *; gdrive_download()
    t = time.time()
    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    os.remove(name) if os.path.exists(name) else None  # remove existing
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system('curl -c ./cookie -s -L "drive.google.com/uc?export=download&id=%s" > %s ' % (id, out))
    if os.path.exists('cookie'):  # large file
        s = 'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm=%s&id=%s" -o %s' % (get_token(), id, name)
    else:  # small file
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    r = os.system(s)  # execute, capture return
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


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""

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
