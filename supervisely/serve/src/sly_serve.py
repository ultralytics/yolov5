import functools
import json
import os
import pathlib
import sys
import traceback

import supervisely as sly


import sly_globals as g
import sly_functions as f

import nn_utils as nn_utils
import sly_apply_nn_to_video as nn_to_video



@g.my_app.callback("inference_batch_ids")
@sly.timeit
@send_error_data
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = []
    for info in infos:
        paths.append(os.path.join(g.my_app.data_dir, sly.rand_str(10) + info.name))
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    results = f.inference_images_dir(img_paths=paths,
                                     context=context,
                                     state=state,
                                     app_logger=app_logger)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


def main():

    f.preprocess()

    g.my_app.run()


