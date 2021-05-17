import supervisely_lib as sly
import sly_metrics as metrics


empty_gallery = {
    "content": {
        "projectMeta": sly.ProjectMeta().to_json(),
        "annotations": {},
        "layout": []
    }
}


def init(data, state):
    _init_start_state(state)
    _init_galleries(data)
    _init_progress(data)
    _init_output(data)
    metrics.init(data, state)


def _init_start_state(state):
    state["started"] = False
    state["activeNames"] = []


def _init_galleries(data):
    data["vis"] = empty_gallery
    data["labelsVis"] = empty_gallery
    data["predVis"] = empty_gallery
    data["syncBindings"] = []


def _init_progress(data):
    data["progressName"] = ""
    data["currentProgress"] = 0
    data["totalProgress"] = 0
    data["currentProgressLabel"] = ""
    data["totalProgressLabel"] = ""


def _init_output(data):
    data["outputUrl"] = ""
    data["outputName"] = ""