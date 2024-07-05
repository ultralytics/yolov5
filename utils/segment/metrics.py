# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Model validation metrics."""

import numpy as np

from ..metrics import ap_per_class


def fitness(x):
    """
    Evaluates the model's fitness using a weighted sum of 8 metrics.
    
    Args:
        x (np.ndarray): A 2D array of shape [N, 8] containing metric values where N is the number of samples.
            The metrics are expected to be in the order: precision, recall, mAP_50, mAP, F1, F2, mAP_75, and mAP_s.
    
    Returns:
        float: The weighted sum of the metrics, computed using predetermined weights. The fitness score is a 
            single floating-point value representing the model's performance, with higher values indicating 
            better fitness.
    
    Example:
        ```python
        metrics = np.array([[0.8, 0.8, 0.6, 0.7, 0.9, 0.5, 0.6, 0.7]])
        score = fitness(metrics)
        print(score)  # Output will depend on the specified weights.
        ```
    
    Note:
        The weights are applied as follows: [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.9], emphasizing the mAP and F1 scores.
    """
    w = [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.9]
    return (x[:, :8] * w).sum(1)


def ap_per_class_box_and_mask(
    tp_m,
    tp_b,
    conf,
    pred_cls,
    target_cls,
    plot=False,
    save_dir=".",
    names=(),
):
    """
    Calculates average precision (AP) metrics for both bounding boxes and masks, distinguishing performance by class.
    
    Args:
        tp_m (np.ndarray): True positives for masks, an array shape [num_detections].
        tp_b (np.ndarray): True positives for bounding boxes, an array shape [num_detections].
        conf (np.ndarray): Confidence scores for detections, an array shape [num_detections].
        pred_cls (np.ndarray): Predicted class indices for detections, an array shape [num_detections].
        target_cls (np.ndarray): True class indices for ground truth, an array shape [num_detections].
        plot (bool): Whether to plot precision-recall curves. Default is False.
        save_dir (str): Directory to save plots if plot=True. Default is current directory ".".
        names (tuple[str, ...]): Optional list of class names for plotting. Default is empty tuple.
    
    Returns:
        dict: Contains precision (p), recall (r), AP (ap), F1-score (f1), and AP by class (ap_class) for boxes and masks.
        The dictionary has the following structure:
        {
            "boxes": {
                "p" (np.ndarray): Precision for each class.
                "r" (np.ndarray): Recall for each class.
                "ap" (np.ndarray): Average precision for each class.
                "f1" (np.ndarray): F1 score for each class.
                "ap_class" (np.ndarray): AP by class.
            },
            "masks": {
                "p" (np.ndarray): Precision for each class.
                "r" (np.ndarray): Recall for each class.
                "ap" (np.ndarray): Average precision for each class.
                "f1" (np.ndarray): F1 score for each class.
                "ap_class" (np.ndarray): AP by class.
            }
        }
    
    Note:
        This function calls `ap_per_class` from the metrics module to compute the required metrics for both boxes and masks.
        For more information on `ap_per_class`, refer to: https://github.com/ultralytics/yolov5
    
    Example:
        ```python
        results = ap_per_class_box_and_mask(tp_m, tp_b, conf, pred_cls, target_cls, plot=False, save_dir=".", names=())
        print(results['boxes']['ap'])  # Print AP for bounding boxes
        print(results['masks']['ap'])  # Print AP for masks
        ```
    """
    results_boxes = ap_per_class(
        tp_b, conf, pred_cls, target_cls, plot=plot, save_dir=save_dir, names=names, prefix="Box"
    )[2:]
    results_masks = ap_per_class(
        tp_m, conf, pred_cls, target_cls, plot=plot, save_dir=save_dir, names=names, prefix="Mask"
    )[2:]

    return {
        "boxes": {
            "p": results_boxes[0],
            "r": results_boxes[1],
            "ap": results_boxes[3],
            "f1": results_boxes[2],
            "ap_class": results_boxes[4],
        },
        "masks": {
            "p": results_masks[0],
            "r": results_masks[1],
            "ap": results_masks[3],
            "f1": results_masks[2],
            "ap_class": results_masks[4],
        },
    }


class Metric:
    def __init__(self) -> None:
        """
        Initializes the Metric class, defining attributes for precision, recall, F1 score, average precision, and class indices.
                
        Attributes:
            p (list[float]): List to store precision values for each class.
            r (list[float]): List to store recall values for each class.
            f1 (list[float]): List to store F1 score values for each class.
            all_ap (list[float]): List to store average precision values for each class across 10 IoU thresholds.
        """
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """
        AP@0.5 of all classes.
        
        Returns:
            List[float]: The Average Precision at IoU threshold 0.5 for each class, represented as a list of floats.
        
        Note:
            This property computes the AP@0.5 using the information stored in `self.all_ap`, which is expected to be a 
            list of AP values at various IoU thresholds. Specifically, it extracts the AP value at index 0, corresponding 
            to the 0.5 IoU threshold.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        AP@0.5:0.95
        
        Returns:
            (np.ndarray | list): Array of average precision values from 0.5 to 0.95 IoU thresholds if available,
            otherwise an empty list. The array length corresponds to the number of classes.
        
        Notes:
            The average precision (AP) is computed at different Intersection over Union (IoU) thresholds, specifically
            from 0.5 to 0.95 with a step size of 0.05. This comprehensive metric aids in evaluating the performance
            of the model across varying degrees of localization accuracy.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Class Metric:
        
            def mp(self) -> float:
                """
                Computes the mean precision (mp) across all classes.
        
                Returns:
                    float: The mean precision value calculated by averaging precision scores across all object classes.
                
                Example:
                    ```python
                    metric = Metric()
                    mean_precision = metric.mp
                    ```
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Summarizes the mean recall of all classes.
        
        Args:
            None
        
        Returns:
            float: The mean recall score across all classes.
        
        Notes:
            The recall is calculated as the fraction of true positive detections out of all actual positives. 
            This is a critical metric for evaluating the performance of object detection models in correctly
            identifying all relevant objects.
        
        Example:
            ```python
            metric = Metric()
            mean_recall = metric.mr
            ```
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Provides the mean Average Precision at IoU=0.5 (mAP@0.5) across all classes.
        
        Returns:
            float: The mean AP@0.5. If no AP data is available, returns 0.0.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Computes the mean Average Precision (mAP) across IoU thresholds from 0.5 to 0.95 for all classes.
        
        Returns:
            float: Mean Average Precision (mAP) across all classes and specified IoU thresholds.
        
        Notes:
            mAP is commonly used in object detection tasks to evaluate the accuracy of a model by averaging the precision-recall curves across different intersection-over-union (IoU) thresholds.
        
        Example:
            ```python
            metric = Metric()
            mean_ap = metric.map
            print(f"Mean AP@0.5:0.95: {mean_ap}")
            ```
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """
        Mean of results, return mp, mr, map50, map.
        
        Returns:
            dict: A dictionary containing the mean precision (mp), mean recall (mr), mean average precision at IoU threshold 0.5 
            (map50), and mean average precision at IoU thresholds from 0.5 to 0.95 (map).
            
        Examples:
            ```python
            metric = Metric()
            results = metric.mean_results()
            print(results)
            # Output: {'mp': 0.75, 'mr': 0.73, 'map50': 0.81, 'map': 0.79}
            ```
        """
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """
        Class-aware result for a specific class index.
        
        Args:
            i (int): Index of the class for which the results are requested.
        
        Returns:
            tuple: A tuple containing the precision (float), recall (float), AP@0.5 (float), and AP@0.5:0.95 (float) for the specified class.
        """
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        """
        Calculates and returns mean Average Precision (mAP) for each class given the number of classes.
        
        Args:
            nc (int): Number of classes.
        
        Returns:
            np.ndarray: An array of shape (nc,) containing the mean Average Precision (mAP) for each class.
        
        Notes:
            This function assumes that the attribute `ap_class_index` is a list of class indices and `ap` is a property 
            that retrieves the AP values for respective classes.
        
        Example:
            ```python
            metric = Metric()
            nc = 20  # Example number of classes
            maps = metric.get_maps(nc)
            print(maps)
            ```
        """
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Updates the metric attributes with the latest set of results.
        
        Args:
            results (tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)): 
                A tuple containing:
                - p (np.ndarray): Precision values for each class, shape (nc,).
                - r (np.ndarray): Recall values for each class, shape (nc,).
                - all_ap (np.ndarray): Average precision values over all thresholds for each class, shape (nc, 10).
                - f1 (np.ndarray): F1 scores for each class, shape (nc,).
                - ap_class_index (np.ndarray): Class indices corresponding to the calculated AP values, shape (nc,).
        
        Returns:
            None
        
        Note:
            The `results` tuple must contain precisely the five elements in the specified order.
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Metric for boxes and masks."""

    def __init__(self) -> None:
        """
        Initializes Metric objects for bounding boxes and masks to compute performance metrics in the Metrics class.
        
        Attributes:
            metric_box (Metric): A Metric object to handle performance metrics related to bounding boxes.
            metric_mask (Metric): A Metric object to handle performance metrics related to masks (optional, based on usage).
        
        Notes:
            This class serves as a container for two Metric objects, facilitating the computation of validation metrics for
            both bounding boxes and instance masks. Each Metric object provides detailed performance metrics such as precision,
            recall, F1 score, and average precision (AP) across various thresholds.
        """
        self.metric_box = Metric()
        self.metric_mask = Metric()

    def update(self, results):
        """
        Update the performance metrics for bounding boxes and masks.
        
        Args:
            results (dict): A dictionary containing two sub-dictionaries under keys 'boxes' and 'masks'. Each sub-
                dictionary should contain performance metrics, specifically precision (p), recall (r), average 
                precision (ap), F1 score (f1), and class indices (ap_class).
        
        Returns:
            None
        
        Example:
            ```python
            results = {
                'boxes': {'p': [...], 'r': [...], 'ap': [...], 'f1': [...], 'ap_class': [...]},
                'masks': {'p': [...], 'r': [...], 'ap': [...], 'f1': [...], 'ap_class': [...]}
            }
            metrics.update(results)
            ```
        """
        self.metric_box.update(list(results["boxes"].values()))
        self.metric_mask.update(list(results["masks"].values()))

    def mean_results(self):
        """
        Computes and returns the mean results for both box and mask metrics as a tuple of their individual means.
        
        Returns:
            tuple: A tuple containing four elements:
                - box_mp (float): Mean precision for boxes.
                - box_mr (float): Mean recall for boxes.
                - box_map50 (float): Mean Average Precision at IoU threshold of 0.5 for boxes.
                - box_map (float): Mean Average Precision at IoU thresholds ranging from 0.5 to 0.95 for boxes.
                - mask_mp (float): Mean precision for masks.
                - mask_mr (float): Mean recall for masks.
                - mask_map50 (float): Mean Average Precision at IoU threshold of 0.5 for masks.
                - mask_map (float): Mean Average Precision at IoU thresholds ranging from 0.5 to 0.95 for masks.
        
        Example:
            ```python
            from ultralytics import Metrics
        
            metrics = Metrics()
            mean_results = metrics.mean_results()
            print(mean_results)  # Prints mean precision, recall, and AP metrics for both boxes and masks
            ```
        """
        return self.metric_box.mean_results() + self.metric_mask.mean_results()

    def class_result(self, i):
        """
        Returns the sum of box and mask metric results for a specified class index.
        
        Args:
            i (int): The class index for which the metric results are to be returned.
        
        Returns:
            tuple: A tuple containing the following metrics for the specified class index:
                - p_box (float): Precision for boxes.
                - r_box (float): Recall for boxes.
                - ap50_box (float): Average Precision at IoU=0.50 for boxes.
                - ap_box (float): Average Precision from IoU=0.50 to IoU=0.95 for boxes.
                - p_mask (float): Precision for masks.
                - r_mask (float): Recall for masks.
                - ap50_mask (float): Average Precision at IoU=0.50 for masks.
                - ap_mask (float): Average Precision from IoU=0.50 to IoU=0.95 for masks.
        
        Example:
            ```python
            metrics = Metrics()
            box_and_mask_results = metrics.class_result(0)
            print(box_and_mask_results)
            ```
        """
        return self.metric_box.class_result(i) + self.metric_mask.class_result(i)

    def get_maps(self, nc):
        """
        Calculates and returns the sum of mean Average Precisions (mAPs) for both box and mask metrics for a specified 
        number of classes.
        
        Args:
            nc (int): The number of classes for which mAP is calculated.
        
        Returns:
            (np.ndarray): An array of mAP values for the specified number of classes, combining box and mask metrics.
        """
        return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)

    @property
    def ap_class_index(self):
        """
        Returns the class index for average precision, shared by both bounding box and mask metrics.
        
        Returns:
            (list[int]): A list of class indices corresponding to average precisions for each class.
        """
        return self.metric_box.ap_class_index


KEYS = [
    "train/box_loss",
    "train/seg_loss",  # train loss
    "train/obj_loss",
    "train/cls_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP_0.5(B)",
    "metrics/mAP_0.5:0.95(B)",  # metrics
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP_0.5(M)",
    "metrics/mAP_0.5:0.95(M)",  # metrics
    "val/box_loss",
    "val/seg_loss",  # val loss
    "val/obj_loss",
    "val/cls_loss",
    "x/lr0",
    "x/lr1",
    "x/lr2",
]

BEST_KEYS = [
    "best/epoch",
    "best/precision(B)",
    "best/recall(B)",
    "best/mAP_0.5(B)",
    "best/mAP_0.5:0.95(B)",
    "best/precision(M)",
    "best/recall(M)",
    "best/mAP_0.5(M)",
    "best/mAP_0.5:0.95(M)",
]
