# Model variant file name or identifier (e.g. 'yolov5su.pt', 'yolov5nu.pt', 'yolov5mu.pt')
MODEL_NAME: str = "yolov5su.pt"

# Total training epochs (1 to 100)
EPOCHS: int = 15

# Batch size per GPU/CPU (4, 8, 16, 32)
BATCH_SIZE: int = 16

# Input image resolution size in pixels (320, 416, 512, 640)
IMG_SIZE: int = 640

# Initial learning rate for SGD/Adam (1e-4 to 0.1)
LR0: float = 0.008

# Final OneCycleLR learning rate fraction of LR0 (0.001 to 0.1)
LRF: float = 0.01

# SGD momentum or Adam beta1 parameter (0.8 to 0.99)
MOMENTUM: float = 0.937

# Optimizer weight decay penalty (0.0 to 0.01)
WEIGHT_DECAY: float = 0.0005

# Warmup epochs count before main schedule (0.0 to 5.0)
WARMUP_EPOCHS: float = 2.0

# Optimizer choice ('auto', 'SGD', 'Adam', 'AdamW', 'RMSProp')
OPTIMIZER: str = "auto"

# HSV-Hue augmentation fraction (0.0 to 0.1)
HSV_H: float = 0.015

# HSV-Saturation augmentation fraction (0.0 to 1.0)
HSV_S: float = 0.7

# HSV-Value augmentation fraction (0.0 to 1.0)
HSV_V: float = 0.4

# Horizontal flip augmentation probability (0.0 to 1.0)
FLIPLR: float = 0.5

# Mosaic augmentation probability (0.0 to 1.0)
MOSAIC: float = 1.0
