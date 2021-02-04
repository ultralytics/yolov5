import os
import supervisely_lib as sly

app = sly.AppService()
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])

api: sly.Api = app.public_api
task_id = app.task_id

local_artifacts_dir = None
remote_artifacts_dir = None