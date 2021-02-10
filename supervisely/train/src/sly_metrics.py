import supervisely_lib as sly
import sly_train_globals as globals


def init_chart(title, names, xs, ys):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({
            "name": name,
            "data": [[px, py] for px, py in zip(x, y)]
        })
    return series


def init(data):
    demo_x = [[], []] #[[1, 2, 3, 4], [2, 4, 6, 8]]
    demo_y = [[], []] #[[10, 15, 13, 17], [16, 5, 11, 9]]
    data["mBox"] = init_chart("Box Loss",
                              names=["train", "val"],
                              xs=demo_x,
                              ys=demo_y)

    data["mObjectness"] = init_chart("Objectness Loss",
                                     names=["train", "val"],
                                     xs=demo_x,
                                     ys=demo_y)

    data["mClassification"] = init_chart("Classification Loss",
                                         names=["train", "val"],
                                         xs=demo_x,
                                         ys=demo_y)

    data["mPR"] = init_chart("Precision / Recall",
                             names=["precision", "recall"],
                             xs=demo_x,
                             ys=demo_y)

    data["mMAP"] = init_chart("mAP",
                              names=["mAP@0.5", "mAP@0.5:0.95"],
                              xs=demo_x,
                              ys=demo_y)
    data["lcOptions"] = {
        "smoothingWeight": 0.6
    }


def send_metrics(epoch, epochs, metrics):
    sly.logger.debug(f"Metrics: epoch {epoch} / {epochs}", extra={"metrics": metrics})

    fields = [
        {"field": "data.mBox[0].data", "payload": [[epoch, metrics["train/box_loss"]]], "append": True},
        {"field": "data.mBox[1].data", "payload": [[epoch, metrics["val/box_loss"]]], "append": True},

        {"field": "data.mObjectness[0].data", "payload": [[epoch, metrics["train/obj_loss"]]], "append": True},
        {"field": "data.mObjectness[1].data", "payload": [[epoch, metrics["val/obj_loss"]]], "append": True},

        {"field": "data.mClassification[0].data", "payload": [[epoch, metrics["train/cls_loss"]]], "append": True},
        {"field": "data.mClassification[1].data", "payload": [[epoch, metrics["val/cls_loss"]]], "append": True},

        {"field": "data.mPR[0].data", "payload": [[epoch, metrics["metrics/precision"]]], "append": True},
        {"field": "data.mPR[1].data", "payload": [[epoch, metrics["metrics/recall"]]], "append": True},

        {"field": "data.mMAP[0].data", "payload": [[epoch, metrics["metrics/mAP_0.5"]]], "append": True},
        {"field": "data.mMAP[1].data", "payload": [[epoch, metrics["metrics/mAP_0.5:0.95"]]], "append": True},
    ]
    globals.api.app.set_fields(globals.task_id, fields)
