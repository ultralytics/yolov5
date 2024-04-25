import cv2
try:
    cv2.setNumThreads(0)
    print("setNumThreads is available.")
except AttributeError:
    print("setNumThreads is not available in the installed cv2 module.")
