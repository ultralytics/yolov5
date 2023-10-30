from datetime import datetime

from yolov5.utils.general import LOGGER


def extract_upload_date(path):
    try:
        # Split the path at the first occurrence of "wd/INPUT"
        parts = path.split("wd/INPUT", 1)
        # Split into two parts, skipping the first Azure input storage part
        date_time_str, image_filename = parts[1].split("/", 2)[1:]
    except Exception as e:
        LOGGER.info(f"Invalid path structure, can not parse path: {path}")
        raise e

    try:
        image_upload_date = datetime.strptime(date_time_str, "%Y-%m-%d_%H:%M:%S")
        image_upload_date = image_upload_date.strftime("%Y-%m-%d")
    except ValueError as e:
        LOGGER.info(f"Invalid folder structure, can not retrieve date: {date_time_str}")
        raise e

    return image_filename, image_upload_date


def get_current_time():
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")  # Format the datetime as a string
    return current_time_str
