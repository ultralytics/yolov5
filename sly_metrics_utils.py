import supervisely_lib as sly

chart_train_style = {
    "name": "train",
    "mode": "lines+markers",
    "line": {
        "color": "rgb(0, 0, 255)",
        "width": 2
    }
}

chart_val_style = {
    "name": "val",
    "mode": "lines+markers",
    "line": {
        "color": "rgb(255, 128, 0)",
        "width": 2
    }
}

chart_layout = {
    "xaxis": {
        # "title": "epoch",
        "automargin": True
    },
    "yaxis": {
        # "title": "value",
        "automargin": True
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.01,
        "xanchor": "right",
        "x": 1
    }
}


def init_chart(title, names, colors, xs, ys):
    data = []
    for name, color, x, y in zip(names, colors, xs, ys):
        data.append({
            "x": x,
            "y": y,
            "name": name,
            "mode": "lines+markers",
            "line": {
                "color": f"rgb({color[0]}, {color[1]}, {color[2]})",
                "width": 2
            }
        })

    chart = {
        "data": data,
        "layout": {
            "title": title,
            **chart_layout
        }
    }
    return chart


def init_metrics(data):
    data["mBox"] = init_chart("Box",
                              names=["train", "val"],
                              colors=[[0, 0, 255], [255, 128, 0]],
                              xs=[[1, 2, 3, 4], [2, 4, 6, 8]],
                              ys=[[10, 15, 13, 17], [16, 5, 11, 9]])

    data["mObjectness"] = init_chart("Objectness",
                                     names=["train", "val"],
                                     colors=[[0, 0, 255], [255, 128, 0]],
                                     xs=[[1, 2, 3, 4], [2, 4, 6, 8]],
                                     ys=[[10, 15, 13, 17], [16, 5, 11, 9]])

    data["mClassification"] = init_chart("Classification",
                                         names=["train", "val"],
                                         colors=[[0, 0, 255], [255, 128, 0]],
                                         xs=[[1, 2, 3, 4], [2, 4, 6, 8]],
                                         ys=[[10, 15, 13, 17], [16, 5, 11, 9]])

    data["mPR"] = init_chart("Precision / Recall",
                             names=["precision", "recall"],
                             colors=[[255, 0, 255], [127, 0, 255]],
                             xs=[[1, 2, 3, 4], [2, 4, 6, 8]],
                             ys=[[10, 15, 13, 17], [16, 5, 11, 9]])

    data["mMAP"] = init_chart("mAP",
                              names=["mAP@0.5", "mAP@0.5:0.95"],
                              colors=[[255, 0, 255], [0, 255, 255]],
                              xs=[[1, 2, 3, 4], [2, 4, 6, 8]],
                              ys=[[10, 15, 13, 17], [16, 5, 11, 9]])
