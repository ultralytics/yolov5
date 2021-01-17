import supervisely_lib as sly

import sys
sys.argv.append('--weights')
sys.argv.append("maxim")

import train

train.main()

x = 10
x += 1

# my_app = sly.AppService()
#
# TEAM_ID = os.environ["context.teamId"]
# WORKSPACE_ID = os.environ["context.workspaceId"]
# INPUT_DIR = os.environ.get("modal.state.slyFolder")
# INPUT_FILE = os.environ.get("modal.state.slyFile")


#@my_app.callback("yolov5_sly_converter")
#@sly.timeit
def yolov5_sly_converter(api: sly.Api, task_id, context, state, app_logger):
    pass


def main():
    # sly.logger.info("Script arguments", extra={
    #     "context.teamId": TEAM_ID,
    #     "context.workspaceId": WORKSPACE_ID,
    #     "modal.state.slyFolder": INPUT_DIR,
    #     "modal.state.slyFile": INPUT_FILE,
    #     "CONFIG_DIR": os.environ.get("CONFIG_DIR", None)
    # })

    #my_app.run(initial_events=[{"command": "yolov5_sly_converter"}])
    pass


if __name__ == "__main__":
    sly.main_wrapper("main", main)