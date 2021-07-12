import GroziPrep as gp
import train


def main():
    gp.etl()
    train.run(imgsz=3264, batch=16, epochs=5, data="grozi.yaml", weights="yolov5s.pt")


if __name__ == "__main__":
    main()
