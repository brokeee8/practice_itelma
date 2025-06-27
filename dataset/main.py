from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")
    model.train(data="dataset/data.yaml", epochs=100, imgsz=640)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
