import os
import tqdm
import boto3
import logging
import traceback


PRODUCT = os.environ["PRODUCT"]
USECASE = os.environ["USECASE"]
BUCKET = os.environ["BUCKET"]

ACCESS_KEY_ID = os.environ["ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = os.environ["SECRET_ACCESS_KEY"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_s3_weights(bucket, object_key, download_path, s3):
    # Get the file size
    file_size = s3.head_object(Bucket=bucket, Key=object_key)['ContentLength']
    # Download the file with a progress bar
    with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=download_path) as pbar:
        s3.download_file(bucket, object_key, download_path,
                         Callback=lambda x: pbar.update(x))


if __name__ == "__main__":

    # Set up the client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY
    )

    # Set the name of the bucket and the file you want to download
    bucket = BUCKET
    object_key = f"{PRODUCT}/{USECASE}/weights.pt"
    download_path = f"model/{USECASE}_weights.pt"
    os.makedirs("model", exist_ok=True)

    # To load all models needed for the PRODUCT, a for loop iterating over each USECASE inside the PRODUCT folder might be added in the future.
    # This code focuses on a particular USECASE inside a PROJECT, e.g. usecase of "health_bill" in the product "health"

    # Download weights
    try:
        print(object_key)
        logger.info(f"Downloading weights: s3://{bucket}/{object_key}")
        download_s3_weights(bucket, object_key, download_path, s3)
        print(f"Successfully downloaded {object_key} into {download_path}")
    except Exception as e:
        print(e)
        print(traceback.format_exc())

        print("Download failed")
