# Ultralytics YOLOv5 🚀, AGPL-3.0 license
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
    """
    Load a ClearML dataset and populate the internal data dictionary with its contents.

    Args:
        clearml_info_string (str): A string containing the ClearML dataset identifier, typically in the format
            'clearml://<dataset_id>'.

    Returns:
        dict: A dictionary containing paths for 'train', 'test', and 'val' datasets along with 'nc' (number of classes)
            and 'names' (class names).

    Raises:
        ValueError: If multiple or no YAML files are found in the dataset root path.
        AssertionError: If essential keys are missing from the YAML file ('train', 'test', 'val', 'nc', 'names').

    Notes:
        This function depends on the ClearML package for dataset retrieval. Ensure ClearML is installed and
        properly configured.

    Example:
        ```python
        from ultralytics import construct_dataset

        clearml_info_string = 'clearml://your-dataset-id'
        dataset_dict = construct_dataset(clearml_info_string)
        print(dataset_dict)
        ```
    """
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

    assert set(
        dataset_definition.keys()
    ).issuperset(
        {"train", "test", "val", "nc", "names"}
    ), "The right keys were not found in the yaml file, make sure it at least has the following keys: ('train', 'test', 'val', 'nc', 'names')"

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
        Initialize the ClearML logging task for experiment tracking and optionally upload dataset version.

        Args:
            opt (namespace): Command-line arguments for the current training run.
            hyp (dict): Hyperparameters for the current training run.

        Returns:
            None
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
        Log scalars or metrics to ClearML experiment tracking.

        Args:
            metrics (dict): Metrics in a dictionary format, e.g., {"metrics/mAP": 0.8, ...}.
            epoch (int): The epoch number corresponding to the current set of metrics.

        Returns:
            None

        Examples:
            ```python
            clearml_logger = ClearmlLogger(opt, hyp)
            metrics = {"metrics/mAP50": 0.8, "metrics/precision": 0.75}
            epoch = 1
            clearml_logger.log_scalars(metrics, epoch)
            ```

        Notes:
            This method logs metrics for the current epoch to the ClearML dashboard, facilitating experiment tracking and
            visualization. Ensure that 'metrics' dictionary keys follow the "title/series" naming convention, e.g.,
            "metrics/mAP50".
        """
        for k, v in metrics.items():
            title, series = k.split("/")
            self.task.get_logger().report_scalar(title, series, v, epoch)

    def log_model(self, model_path, model_name, epoch=0):
        """
        Log model weights to ClearML.

        Args:
            model_path (Path | str): Path to the model weights file.
            model_name (str): Name of the model to be displayed in ClearML.
            epoch (int, optional): Epoch number representing the state of the model. Defaults to 0.

        Returns:
            None

        Notes:
            This method updates the output model in ClearML with the specified path and name. The model weights file is not
            automatically deleted after uploading, ensuring the file remains accessible on the local system.
        """
        self.task.update_output_model(
            model_path=str(model_path), name=model_name, iteration=epoch, auto_delete_file=False
        )

    def log_summary(self, metrics):
        """
        Log final summary metrics to ClearML.

        Args:
            metrics (dict): Dictionary containing final metrics. Example: {"metrics/mAP": 0.8, "metrics/precision": 0.9, ...}

        Returns:
            None

        Examples:
            ```python
            logger = ClearmlLogger(opt, hyp)
            final_metrics = {"metrics/mAP": 0.85, "metrics/precision": 0.92}
            logger.log_summary(final_metrics)
            ```
        """
        for k, v in metrics.items():
            self.task.get_logger().report_single_value(k, v)

    def log_plot(self, title, plot_path):
        """
        Log image as a plot in the plot section of ClearML.

        Args:
            title (str): Title of the plot.
            plot_path (PosixPath | str): Path to the saved image file.

        Returns:
            None

        Notes:
            This function loads the image from the specified path and displays it using Matplotlib.
            The displayed plot is then logged to ClearML, providing a visual representation within the ClearML UI.

        Examples:
            ```python
            from pathlib import Path
            logger = ClearmlLogger(opt, hyp)
            logger.log_plot("Sample Plot", Path("path/to/plot.png"))
            ```
        """
        img = mpimg.imread(plot_path)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # no ticks
        ax.imshow(img)

        self.task.get_logger().report_matplotlib_figure(title, "", figure=fig, report_interactive=False)

    def log_debug_samples(self, files, title="Debug Samples"):
        """
        Log files (images) as debug samples in the ClearML task.

        Args:
          files (List[PosixPath]): A list of file paths to the images to log.
          title (str, optional): A title that groups together images with the same values. Defaults to "Debug Samples".

        Returns:
          None
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
        Clear and report the bounding boxes on an image to ClearML as a debug sample.

        Args:
            image_path (PosixPath): The path to the original image file.
            boxes (list): List of scaled predictions in the format [xmin, ymin, xmax, ymax, confidence, class].
            class_names (dict): Mapping of class indices to class names.
            image (torch.Tensor): Tensor containing the actual image data.
            conf_threshold (float, optional): Confidence threshold for displaying bounding boxes. Defaults to 0.25.

        Returns:
            None

        Notes:
            This method logs at most `max_imgs_to_log_per_epoch` images per epoch, ensuring that logs are not
            overwhelming. Bounding boxes are only logged if current epoch is divisible by `bbox_interval`
            and the image hasn't already been logged. All images are converted to numpy arrays before annotation.
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
