import os
from pathlib import Path
import sys
import yaml
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


root_source_path = str(Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

# script_path = str(Path(sys.argv[0]).parents[3]))
# root_app_dir = script_path.parent.parent.absolute()
# sly.logger.info(f"Root app directory: {root_app_dir}")
# sys.path.append(root_app_dir)


def init_project_info_and_meta():
    global project_info, project_meta
    project_info = api.project.get_info_by_id(project_id)
    project_meta_json = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta_json)


with open(os.path.join(root_source_path, "data/hyp.scratch.yaml"), 'r') as file:
    scratch_str = file.read()  # yaml.safe_load(

with open(os.path.join(root_source_path, "data/hyp.finetune.yaml"), 'r') as file:
    finetune_str = file.read()  # yaml.safe_load(