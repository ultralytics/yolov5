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
project_info = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))


root_source_dir = str(Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"Source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

with open(os.path.join(root_source_dir, "data/hyp.scratch.yaml"), 'r') as file:
    scratch_str = file.read()  # yaml.safe_load(

with open(os.path.join(root_source_dir, "data/hyp.finetune.yaml"), 'r') as file:
    finetune_str = file.read()  # yaml.safe_load(


runs_dir = os.path.join(my_app.data_dir, 'runs')
sly.fs.mkdir(runs_dir, remove_content_if_exists=True)  # for debug, does nothing in production
experiment_name = str(task_id)
local_artifacts_dir = os.path.join(runs_dir, experiment_name)
sly.logger.info(f"All training artifacts will be saved to local directory {local_artifacts_dir}")
remote_artifacts_dir = os.path.join("/yolov5_train", project_info.name, experiment_name)
remote_artifacts_dir = api.file.get_free_dir_name(team_id, remote_artifacts_dir)
sly.logger.info(f"After training artifacts will be uploaded to Team Files: {remote_artifacts_dir}")