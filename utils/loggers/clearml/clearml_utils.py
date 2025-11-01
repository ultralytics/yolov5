# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Main Logger class for ClearML experiment tracking."""

import glob
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics.utils.plotting import Annotator, colors

try:
    import clearml
    from clearml import Dataset, Task

    assert hasattr(clearml, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    clearml = None


def construct_dataset(clearml_info_string):
    """Load in a clearml dataset and fill the internal data_dict with its contents."""
    dataset_id = clearml_info_string.replace("clearml://", "")
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_root_path = Path(dataset.get_local_copy())

    # We'll search for the yaml file definition in the dataset
    yaml_filenames = list(glob.glob(str(dataset_root_path / "*.yaml")) + glob.glob(str(dataset_root_path / "*.yml")))
    if len(yaml_filenames) > 1:
        raise ValueError(
            "More than one yaml file was found in the dataset root, cannot determine which one contains "
            "the dataset definition this way."
        )
    elif not yaml_filenames:
        raise ValueError(
            "No yaml definition found in dataset root path, check that there is a correct yaml file "
            "inside the dataset root path."
        )
    with open(yaml_filenames[0]) as f:
        dataset_definition = yaml.safe_load(f)

    assert set(dataset_definition.keys()).issuperset({"train", "test", "val", "nc", "names"}), (
        "The right keys were not found in the yaml file, make sure it at least has the following keys: ('train', 'test', 'val', 'nc', 'names')"
    )

    data_dict = {
        "train": (
            str((dataset_root_path / dataset_definition["train"]).resolve()) if dataset_definition["train"] else None
        )
    }
    data_dict["test"] = (
        str((dataset_root_path / dataset_definition["test"]).resolve()) if dataset_definition["test"] else None
    )
    data_dict["val"] = (
        str((dataset_root_path / dataset_definition["val"]).resolve()) if dataset_definition["val"] else None
    )
    data_dict["nc"] = dataset_definition["nc"]
    data_dict["names"] = dataset_definition["names"]

    return data_dict


class ClearmlLogger:
    """
    Log training runs, datasets, models, and predictions to ClearML.

    This logger sends information to ClearML at app.clear.ml or to your own hosted server. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics, code information and basic data metrics
    and analyses.

    By providing additional command line arguments to train.py, datasets, models and predictions can also be logged.
    """

    def __init__(self, opt, hyp):
        """
        - Initialize ClearML Task, this object will capture the experiment
        - Upload dataset version to ClearML Data if opt.upload_dataset is True.

        Arguments:
        opt (namespace) -- Commandline arguments for this run
        hyp (dict) -- Hyperparameters for this run
        """
        self.current_epoch = 0
        # Keep tracked of amount of logged images to enforce a limit
        self.current_epoch_logged_images = set()
        # Maximum number of images to log to clearML per epoch
        self.max_imgs_to_log_per_epoch = 16
        # Get the interval of epochs when bounding box images should be logged
        # Only for detection task though!
        if "bbox_interval" in opt:
            self.bbox_interval = opt.bbox_interval
        self.clearml = clearml
        self.task = None
        self.data_dict = None
        if self.clearml:
            self.task = Task.init(
                project_name="YOLOv5" if str(opt.project).startswith("runs/") else opt.project,
                task_name=opt.name if opt.name != "exp" else "Training",
                tags=["YOLOv5"],
                output_uri=True,
                reuse_last_task_id=opt.exist_ok,
                auto_connect_frameworks={"pytorch": False, "matplotlib": False},
                # We disconnect pytorch auto-detection, because we added manual model save points in the code
            )
            # ClearML's hooks will already grab all general parameters
            # Only the hyperparameters coming from the yaml config file
            # will have to be added manually!
            self.task.connect(hyp, name="Hyperparameters")
            self.task.connect(opt, name="Args")

            # Make sure the code is easily remotely runnable by setting the docker image to use by the remote agent
            self.task.set_base_docker(
                "ultralytics/yolov5:latest",
                docker_arguments='--ipc=host -e="CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1"',
                docker_setup_bash_script="pip install clearml",
            )

            # Get ClearML Dataset Version if requested
            if opt.data.startswith("clearml://"):
                # data_dict should have the following keys:
                # names, nc (number of classes), test, train, val (all three relative paths to ../datasets)
                self.data_dict = construct_dataset(opt.data)
                # Set data to data_dict because wandb will crash without this information and opt is the best way
                # to give it to them
                opt.data = self.data_dict

    def log_scalars(self, metrics, epoch):
        """
        Log scalars/metrics to ClearML.

        Arguments:
        metrics (dict) Metrics in dict format: {"metrics/mAP": 0.8, ...}
        epoch (int) iteration number for the current set of metrics
        """
        for k, v in metrics.items():
            title, series = k.split("/")
            self.task.get_logger().report_scalar(title, series, v, epoch)

    def log_model(self, model_path, model_name, epoch=0):
        """
        Log model weights to ClearML.

        Arguments:
        model_path (PosixPath or str) Path to the model weights
        model_name (str) Name of the model visible in ClearML
        epoch (int) Iteration / epoch of the model weights
        """
        self.task.update_output_model(
            model_path=str(model_path), name=model_name, iteration=epoch, auto_delete_file=False
        )

    def log_summary(self, metrics):
        """
        Log final metrics to a summary table.

        Arguments:
        metrics (dict) Metrics in dict format: {"metrics/mAP": 0.8, ...}
        """
        for k, v in metrics.items():
            self.task.get_logger().report_single_value(k, v)

    def log_plot(self, title, plot_path):
        """
        Log image as plot in the plot section of ClearML.

        Arguments:
        title (str) Title of the plot
        plot_path (PosixPath or str) Path to the saved image file
        """
        img = mpimg.imread(plot_path)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # no ticks
        ax.imshow(img)

        self.task.get_logger().report_matplotlib_figure(title, "", figure=fig, report_interactive=False)

    def log_debug_samples(self, files, title="Debug Samples"):
        """
        Log files (images) as debug samples in the ClearML task.

        Arguments:
        files (List(PosixPath)) a list of file paths in PosixPath format
        title (str) A title that groups together images with the same values
        """
        for f in files:
            if f.exists():
                it = re.search(r"_batch(\d+)", f.name)
                iteration = int(it.groups()[0]) if it else 0
                self.task.get_logger().report_image(
                    title=title, series=f.name.replace(f"_batch{iteration}", ""), local_path=str(f), iteration=iteration
                )

    def log_image_with_boxes(self, image_path, boxes, class_names, image, conf_threshold=0.25):
        """
        Draw the bounding boxes on a single image and report the result as a ClearML debug sample.

        Arguments:
        image_path (PosixPath) the path the original image file
        boxes (list): list of scaled predictions in the format - [xmin, ymin, xmax, ymax, confidence, class]
        class_names (dict): dict containing mapping of class int to class name
        image (Tensor): A torch tensor containing the actual image data
        """
        if (
            len(self.current_epoch_logged_images) < self.max_imgs_to_log_per_epoch
            and self.current_epoch >= 0
            and (self.current_epoch % self.bbox_interval == 0 and image_path not in self.current_epoch_logged_images)
        ):
            im = np.ascontiguousarray(np.moveaxis(image.mul(255).clamp(0, 255).byte().cpu().numpy(), 0, 2))
            annotator = Annotator(im=im, pil=True)
            for i, (conf, class_nr, box) in enumerate(zip(boxes[:, 4], boxes[:, 5], boxes[:, :4])):
                color = colors(i)

                class_name = class_names[int(class_nr)]
                confidence_percentage = round(float(conf) * 100, 2)
                label = f"{class_name}: {confidence_percentage}%"

                if conf > conf_threshold:
                    annotator.rectangle(box.cpu().numpy(), outline=color)
                    annotator.box_label(box.cpu().numpy(), label=label, color=color)

            annotated_image = annotator.result()
            self.task.get_logger().report_image(
                title="Bounding Boxes", series=image_path.name, iteration=self.current_epoch, image=annotated_image
            )
            self.current_epoch_logged_images.add(image_path)
