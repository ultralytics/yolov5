import supervisely_lib as sly


def download_project(api:sly.Api, directory, cache=None):
    sly.download_project(api, directory)