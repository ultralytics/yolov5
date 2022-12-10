import os
import supervisely as sly
from utils.torch_utils import select_device
from supervisely.app.v1.app_service import AppService

my_app: AppService = AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
TASK_ID = int(os.environ['TASK_ID'])
customWeightsPath = os.environ['modal.state.slyFile']
device = select_device(device='cpu')
image_size = 640
ts = None
batch_size = 1
grid = True
args = dict(my_app=my_app, TEAM_ID=TEAM_ID)
