import supervisely_lib as sly
import sly_train_globals as globals


def init_chart(title, names, xs, ys, smoothing=None):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({
            "name": name,
            "data": [[px, py] for px, py in zip(x, y)]
        })
    result = {
        "options": {
            "title": title,
            #"groupKey": "my-synced-charts",
        },
        "series": series
    }
    if smoothing is not None:
        result["options"]["smoothingWeight"] = smoothing
    return result


def init(data, state):
    demo_x = [[], []] #[[1, 2, 3, 4], [2, 4, 6, 8]]
    demo_y = [[], []] #[[10, 15, 13, 17], [16, 5, 11, 9]]
    data["mGIoU"] = init_chart("GIoU",
                               names=["train", "val"],
                               xs=demo_x,
                               ys=demo_y,
                               smoothing=0.6)

    data["mObjectness"] = init_chart("Objectness",
                                     names=["train", "val"],
                                     xs=demo_x,
                                     ys=demo_y,
                                     smoothing=0.6)

    data["mClassification"] = init_chart("Classification",
                                         names=["train", "val"],
                                         xs=demo_x,
                                         ys=demo_y,
                                         smoothing=0.6)

    data["mPR"] = init_chart("Pr + Rec",
                             names=["precision", "recall"],
                             xs=demo_x,
                             ys=demo_y)

    data["mMAP"] = init_chart("mAP",
                              names=["mAP@0.5", "mAP@0.5:0.95"],
                              xs=demo_x,
                              ys=demo_y)
    state["smoothing"] = 0.6


def send_metrics(epoch, epochs, metrics, log_period=1):
    sly.logger.debug(f"Metrics: epoch {epoch + 1} / {epochs}", extra={"metrics": metrics})

    if epoch % log_period == 0 or epoch + 1 == epochs:
        fields = [
            {"field": "data.mGIoU.series[0].data", "payload": [[epoch, metrics["train/box_loss"]]], "append": True},
            {"field": "data.mGIoU.series[1].data", "payload": [[epoch, metrics["val/box_loss"]]], "append": True},

            {"field": "data.mObjectness.series[0].data", "payload": [[epoch, metrics["train/obj_loss"]]], "append": True},
            {"field": "data.mObjectness.series[1].data", "payload": [[epoch, metrics["val/obj_loss"]]], "append": True},

            {"field": "data.mClassification.series[0].data", "payload": [[epoch, metrics["train/cls_loss"]]], "append": True},
            {"field": "data.mClassification.series[1].data", "payload": [[epoch, metrics["val/cls_loss"]]], "append": True},

            {"field": "data.mPR.series[0].data", "payload": [[epoch, metrics["metrics/precision"]]], "append": True},
            {"field": "data.mPR.series[1].data", "payload": [[epoch, metrics["metrics/recall"]]], "append": True},

            {"field": "data.mMAP.series[0].data", "payload": [[epoch, metrics["metrics/mAP_0.5"]]], "append": True},
            {"field": "data.mMAP.series[1].data", "payload": [[epoch, metrics["metrics/mAP_0.5:0.95"]]], "append": True},
        ]
        globals.api.app.set_fields(globals.task_id, fields)
