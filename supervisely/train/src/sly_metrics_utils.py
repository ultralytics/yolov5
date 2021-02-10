# old plotly implementation
# =================================================================
# import supervisely_lib as sly
# import sly_train_globals as globals
#
#
# chart_train_style = {
#     "name": "train",
#     "mode": "lines+markers",
#     "line": {
#         "color": "rgb(0, 0, 255)",
#         "width": 2
#     }
# }
#
# chart_val_style = {
#     "name": "val",
#     "mode": "lines+markers",
#     "line": {
#         "color": "rgb(255, 128, 0)",
#         "width": 2
#     }
# }
#
# chart_layout = {
#     "xaxis": {
#         # "title": "epoch",
#         "automargin": True
#     },
#     "yaxis": {
#         # "title": "value",
#         "automargin": True
#     },
#     "legend": {
#         "orientation": "h",
#         "yanchor": "bottom",
#         "y": 0.99,
#         "xanchor": "right",
#         "x": 1
#     }
# }
#
#
# def init_chart(title, names, colors, xs, ys):
#     data = []
#     for name, color, x, y in zip(names, colors, xs, ys):
#         data.append({
#             "x": x,
#             "y": y,
#             "name": name,
#             "mode": "lines+markers",
#             #"type": "scattergl",
#             "line": {
#                 "color": f"rgb({color[0]}, {color[1]}, {color[2]})",
#                 "width": 2
#             }
#         })
#
#     chart = {
#         "data": data,
#         "layout": {
#             "title": {
#                 "text": f"<b>{title}</b>",
#                 "xanchor": "left",
#                 'y': 0.97,
#                 'x': 0.03,
#                 "font": {
#                     "size": 14,
#                     "color": "rgb(96, 96, 96)",
#                     #"color": "rgb(0, 150, 0)",
#                 }
#             },
#             **chart_layout
#         }
#     }
#     return chart
#
#
# def init(data):
#     demo_x = [[], []] #[[1, 2, 3, 4], [2, 4, 6, 8]]
#     demo_y = [[], []] #[[10, 15, 13, 17], [16, 5, 11, 9]]
#     data["mBox"] = init_chart("Box Loss",
#                               names=["train", "val"],
#                               colors=[[0, 0, 255], [255, 128, 0]],
#                               xs=demo_x,
#                               ys=demo_y)
#
#     data["mObjectness"] = init_chart("Objectness Loss",
#                                      names=["train", "val"],
#                                      colors=[[0, 0, 255], [255, 128, 0]],
#                                      xs=demo_x,
#                                      ys=demo_y)
#
#     data["mClassification"] = init_chart("Classification Loss",
#                                          names=["train", "val"],
#                                          colors=[[0, 0, 255], [255, 128, 0]],
#                                          xs=demo_x,
#                                          ys=demo_y)
#
#     data["mPR"] = init_chart("Precision / Recall",
#                              names=["precision", "recall"],
#                              colors=[[255, 0, 255], [127, 0, 255]],
#                              xs=demo_x,
#                              ys=demo_y)
#
#     data["mMAP"] = init_chart("mAP",
#                               names=["mAP@0.5", "mAP@0.5:0.95"],
#                               colors=[[255, 0, 255], [0, 255, 255]],
#                               xs=demo_x,
#                               ys=demo_y)
#
#
# def send_metrics(epoch, epochs, metrics):
#     sly.logger.debug(f"Metrics: epoch {epoch} / {epochs}", extra={"metrics": metrics})
#
#     fields = [
#         {"field": "data.mBox.data[0].x", "payload": epoch, "append": True},
#         {"field": "data.mBox.data[1].x", "payload": epoch, "append": True},
#         {"field": "data.mBox.data[0].y", "payload": metrics["train/box_loss"], "append": True},
#         {"field": "data.mBox.data[1].y", "payload": metrics["val/box_loss"], "append": True},
#
#         {"field": "data.mObjectness.data[0].x", "payload": epoch, "append": True},
#         {"field": "data.mObjectness.data[1].x", "payload": epoch, "append": True},
#         {"field": "data.mObjectness.data[0].y", "payload": metrics["train/obj_loss"], "append": True},
#         {"field": "data.mObjectness.data[1].y", "payload": metrics["val/obj_loss"], "append": True},
#
#         {"field": "data.mClassification.data[0].x", "payload": epoch, "append": True},
#         {"field": "data.mClassification.data[1].x", "payload": epoch, "append": True},
#         {"field": "data.mClassification.data[0].y", "payload": metrics["train/cls_loss"], "append": True},
#         {"field": "data.mClassification.data[1].y", "payload": metrics["val/cls_loss"], "append": True},
#
#         {"field": "data.mPR.data[0].x", "payload": epoch, "append": True},
#         {"field": "data.mPR.data[1].x", "payload": epoch, "append": True},
#         {"field": "data.mPR.data[0].y", "payload": metrics["metrics/precision"], "append": True},
#         {"field": "data.mPR.data[1].y", "payload": metrics["metrics/recall"], "append": True},
#
#         {"field": "data.mMAP.data[0].x", "payload": epoch, "append": True},
#         {"field": "data.mMAP.data[1].x", "payload": epoch, "append": True},
#         {"field": "data.mMAP.data[0].y", "payload": metrics["metrics/mAP_0.5"], "append": True},
#         {"field": "data.mMAP.data[1].y", "payload": metrics["metrics/mAP_0.5:0.95"], "append": True},
#     ]
#     globals.api.app.set_fields(globals.task_id, fields)
