from functools import partial
import os
import time
import supervisely_lib as sly
import sly_train_globals as globals


def update_progress(count, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done_report(count)
    _update_progress_ui(api, task_id, progress)


def _update_progress_ui(api: sly.Api, task_id, progress: sly.Progress, stdout_print=False):
    if progress.need_report():
        fields = [
            {"field": "data.progressName", "payload": progress.message},
            {"field": "data.currentProgressLabel", "payload": progress.current_label},
            {"field": "data.totalProgressLabel", "payload": progress.total_label},
            {"field": "data.currentProgress", "payload": progress.current},
            {"field": "data.totalProgress", "payload": progress.total},
        ]
        api.app.set_fields(task_id, fields)
        if stdout_print is True:
            progress.report_if_needed()


def get_progress_cb(message, total, is_size=False):
    progress = sly.Progress(message, total, is_size=is_size)
    progress_cb = partial(update_progress, api=globals.api, task_id=globals.task_id, progress=progress)
    progress_cb(0)
    return progress_cb


def update_uploading_progress(count, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done(count - progress.current)
    _update_progress_ui(api, task_id, progress, stdout_print=True)


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    name = "open_app.lnk"
    local_path = os.path.join(local_dir, name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts(local_dir, remote_dir):
    def _gen_message(current, total):
        return f"Upload artifacts to Team Files [{current}/{total}] "

    _save_link_to_ui(local_dir, globals.my_app.app_url)
    local_files = sly.fs.list_files_recursively(local_dir)
    total_size = sum([sly.fs.get_file_size(file_path) for file_path in local_files])

    progress = sly.Progress(_gen_message(0, len(local_files)), total_size, is_size=True)
    progress_cb = partial(update_uploading_progress, api=globals.api, task_id=globals.task_id, progress=progress)
    progress_cb(0)

    for idx, local_path in enumerate(local_files):
        remote_path = os.path.join(remote_dir, local_path.replace(local_dir, '').lstrip("/"))
        if globals.api.file.exists(globals.team_id, remote_path):
            progress.iters_done_report(sly.fs.get_file_size(local_path))
        else:
            progress_last = progress.current
            globals.api.file.upload(globals.team_id, local_path, remote_path,
                                    lambda monitor: progress_cb(progress_last + monitor.bytes_read))
        progress.message = _gen_message(idx + 1, len(local_files))
        time.sleep(0.5)