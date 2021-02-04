from functools import partial
import supervisely_lib as sly
import sly_train_globals as globals


def update_progress(count, api: sly.Api, task_id, progress: sly.Progress):
    _update_progress_log(count, progress)
    _update_progress_ui(api, task_id, progress)


def _update_progress_log(count, progress: sly.Progress):
    progress.iters_done_report(count)


def _update_progress_ui(api: sly.Api, task_id, progress: sly.Progress):
    if progress.need_report():
        fields = [
            {"field": "data.progressName", "payload": progress.message},
            {"field": "data.currentProgressLabel", "payload": progress.current_label},
            {"field": "data.totalProgressLabel", "payload": progress.total_label},
            {"field": "data.currentProgress", "payload": progress.current},
            {"field": "data.totalProgress", "payload": progress.total},
        ]
        api.app.set_fields(task_id, fields)


def get_progress_cb(message, total):
    progress = sly.Progress(message, total)
    progress_cb = partial(update_progress, api=globals.api, task_id=globals.task_id, progress=progress)
    progress_cb(0)
    return progress_cb


# from functools import partial
# def func(a, b, progress):
#     print("a = ", a)
#     print("b = ", b)
#     progress.iters_done_report(1)


# def debug_download_progress():
#     team_id = 7
#     remote_path = "/yolov5_train/coco128_002/2390/weights/best.pt"
#     sly.fs.ensure_base_path(remote_path)
#     sly.fs.silent_remove(remote_path)
#     #my_app.public_api.file.download(team_id, remote_path, remote_path, my_app.cache, progress_cb)
#
#     file_info = my_app.public_api.file.get_info_by_path(team_id, remote_path)
#     progress = sly.Progress("doing", file_info.sizeb, is_size=True)
#     progress_cb = partial(update_progress, api=my_app.public_api, task_id=my_app.task_id, progress=progress)
#     my_app.public_api.file.download(team_id, remote_path, remote_path, None, progress_cb)
#     print(sly.fs.file_exists(remote_path))
#     pass