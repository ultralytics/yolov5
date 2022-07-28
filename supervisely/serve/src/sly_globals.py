import os
import sys
import pathlib
import supervisely as sly
from supervisely.app.v1.app_service import AppService

# from dotenv import load_dotenv

import yaml

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)
sys.path.append(str(pathlib.Path(sys.argv[0]).parents[2]))  # supervisely folder
print(str(pathlib.Path(sys.argv[0]).parents[2]))

# load_dotenv(os.path.join(root_source_path, "supervisely", "serve", "debug.env"))
# load_dotenv(os.path.join(root_source_path, "supervisely", "serve", "secret_debug.env"), override=True)

my_app: AppService = AppService()
api = my_app.public_api
task_id = my_app.task_id

logger = sly.logger

sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

meta: sly.ProjectMeta = None

modelWeightsOptions = os.environ['modal.state.modelWeightsOptions']
pretrained_weights = os.environ['modal.state.selectedModel'].lower()
custom_weights = os.environ['modal.state.weightsPath']


DEVICE_STR = os.environ['modal.state.device']
final_weights = None
model = None
half = None
device = None
imgsz = None
stride = None


settings_path = os.path.join(root_source_path, "supervisely/serve/custom_settings.yaml")
sly.logger.info(f"Custom inference settings path: {settings_path}")
with open(settings_path, 'r') as file:
    default_settings_str = file.read()
    default_settings = yaml.safe_load(default_settings_str)



