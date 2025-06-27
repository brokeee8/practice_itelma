from ultralytics import YOLO

model = YOLO("C:/Users/Vlad/runs/detect/train11/weights/best.pt")

results = model("D:/itelma practice/test", save=True)

for r in results:
    r.show()
