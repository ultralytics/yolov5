import os
import sys
import supervisely_lib as sly


#sys.argv.append('--weights')
#sys.argv.append("maxim")
#import train
#train.main()

my_app = sly.AppService()


TEAM_ID = os.environ["context.teamId"]
WORKSPACE_ID = os.environ["context.workspaceId"]


#@my_app.callback("yolov5_sly_converter")
#@sly.timeit
# def yolov5_sly_converter(api: sly.Api, task_id, context, state, app_logger):
#     pass


def main():
    # sly.logger.info("Script arguments", extra={
    #     "context.teamId": TEAM_ID,
    #     "context.workspaceId": WORKSPACE_ID,
    #     "modal.state.slyFolder": INPUT_DIR,
    #     "modal.state.slyFile": INPUT_FILE,
    #     "CONFIG_DIR": os.environ.get("CONFIG_DIR", None)
    # })

    data = {}
    state = {}

    #my_app.run(initial_events=[{"command": "yolov5_sly_converter"}])
    template_path = os.path.join(os.path.dirname(sys.argv[0]), 'supervisely/train/src/gui.html')
    my_app.run(template_path, data, state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)