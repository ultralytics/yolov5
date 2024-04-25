import cv2

def find_available_cameras(limit=10):
    available_cameras = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at source number: {i}")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"No camera found at source number: {i}")
    return available_cameras

if __name__ == "__main__":
    cameras = find_available_cameras()
    print("Available cameras:", cameras)