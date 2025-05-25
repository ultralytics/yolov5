from models.yolo import DetectRAYOLO
from utils.general import LOGGER
from utils.torch_utils import de_parallel

ALL_5_HEADS_MODULES = set(range(41))  # All modules from 0 to 43 (inclusive)

# 4 heads (P2, P3, P4, P5) configuration
# Ranges: (0,8) + (14,37)
FOUR_HEADS_MODULES = set(range(0, 9))  # Includes modules 0 through 8
FOUR_HEADS_MODULES.update(range(14, 38))  # Includes modules 14 through 37

# Lowest config (3 heads: P2, P3, P4) configuration
# Ranges: (0,6) + (18,34)
THREE_HEADS_MODULES = set(range(0, 7))  # Includes modules 0 through 6
THREE_HEADS_MODULES.update(range(18, 35))  # Includes modules 18 through 34


def set_model_grad_status(img, model_instance, heights=[810, 1620]):
    model_unwrapped = de_parallel(model_instance)  # Get the base model if it's DDP-wrapped

    active_module_indices = ALL_5_HEADS_MODULES
    print(img.shape)

    H = img.shape[2]
    H4, H5 = heights
    if H > H5:
        active_module_indices = ALL_5_HEADS_MODULES
    if H5 >= H and H >= H4:
        active_module_indices = FOUR_HEADS_MODULES
    if H4 > H:
        active_module_indices = THREE_HEADS_MODULES

    # 1. Freeze all parameters by default
    for param in model_unwrapped.parameters():
        param.requires_grad = False

    # 2. Unfreeze only the parameters of modules in our 'active_module_indices' set
    for i, m in enumerate(model_unwrapped.model):  # Iterate through the sequential list of modules
        if i in active_module_indices:
            for param in m.parameters():
                param.requires_grad = True

    # 3. Special handling for the Detect layer's internal convolutional heads (self.m)
    # The 'Detect' module itself is typically the last module in model.model.
    # We need to unfreeze its *internal* convs based on which Detect heads are selected.
    if hasattr(model_unwrapped, "detect_layer") and isinstance(model_unwrapped.detect_layer, DetectRAYOLO):
        # Mapping from the *model.model* index that feeds into Detect
        # to the *Detect layer's internal head index* (0 to 4)
        # This mapping is based on your yolov5l_p2_p6_extended.yaml's Detect 'from' indices:
        # Detect from: [[27, 30, 33, 36, 39]]
        detect_input_to_head_idx = {27: 0, 30: 1, 33: 2, 36: 3, 39: 4}

        active_detect_heads = []
        for input_idx_in_model, head_idx_in_detect in detect_input_to_head_idx.items():
            if input_idx_in_model in active_module_indices:  # If the input feature map's source is active
                active_detect_heads.append(head_idx_in_detect)

        for i, conv_head in enumerate(model_unwrapped.detect_layer.m):
            for param in conv_head.parameters():
                if i in active_detect_heads:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    trainable_params_count = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    total_params_count = sum(p.numel() for p in model_instance.parameters())
    print(
        f"Trainable params: {trainable_params_count}/{total_params_count} ({trainable_params_count / total_params_count:.1%})"
    )
    LOGGER.info(
        f"Trainable params: {trainable_params_count}/{total_params_count} ({trainable_params_count / total_params_count:.1%})"
    )
