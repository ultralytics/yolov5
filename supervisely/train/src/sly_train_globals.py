import os
import supervisely_lib as sly

my_app = sly.AppService()
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

api: sly.Api = my_app.public_api
task_id = my_app.task_id

local_artifacts_dir = None
remote_artifacts_dir = None

project_info = None
project_meta = None


def init_project_info_and_meta():
    global project_info, project_meta
    project_info = api.project.get_info_by_id(project_id)
    project_meta_json = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta_json)