import json
import os

import supervisely as sly


def main():
    api = sly.Api.from_env()

    # task id of the deployed model

    task_id = int(os.getenv('TASK_ID'))
    # get information about model
    info = api.task.send_request(task_id, "get_session_info", data={}, timeout=3)
    print("Information about deployed model:")
    print(json.dumps(info, indent=4))

    # get predicted annotations by video id
    video_id = 3013731
    predictions = api.task.send_request(task_id, "inference_video_id",
                                        data={'videoId': video_id,
                                              'startFrameIndex': 0,
                                              'framesCount': 10,
                                              'settings': {
                                                  'conf_thres': 0.4
                                              }}, timeout=60 * 60 * 24)

    print(f"Predictions for {video_id = }")
    print(json.dumps(predictions, indent=4))


if __name__ == "__main__":
    main()
