from datetime import datetime

from utils.general import LOGGER


def parse_image_path(path):
    parts = path.split('/')
    image_filename = parts[-1]
    date_time_str = parts[-2]

    try:
        image_upload_date = datetime.strptime(date_time_str, "%Y-%m-%d_%H:%M:%S")
        image_upload_date = image_upload_date.strftime("%Y-%m-%d")
    except ValueError as e:
        LOGGER.info(f"Invalid folder structure, can not retrieve date: {image_upload_date}")
        raise e

    return image_filename, image_upload_date
