import os
import sly_train_globals as g


def init(data):
    data["outputUrl"] = None
    data["outputName"] = None


def set_task_output():
    file_info = g.api.file.get_info_by_path(g.team_id, os.path.join(g.remote_artifacts_dir, 'results.png'))
    fields = [
        {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
        {"field": "data.outputName", "payload": g.remote_artifacts_dir},
    ]
    g.api.app.set_fields(g.task_id, fields)
    g.api.task.set_output_directory(g.task_id, file_info.id, g.remote_artifacts_dir)