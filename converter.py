import json
import os
from pathlib import Path

def coco_to_yolo(json_path, out_dir, img_dir):
    with open(json_path) as f:
        data = json.load(f)

    categories = {cat['id']: i for i, cat in enumerate(data['categories'])}
    images = {img['id']: img for img in data['images']}

    yolo_labels = Path(out_dir)
    yolo_labels.mkdir(parents=True, exist_ok=True)

    for ann in data['annotations']:
        img_info = images[ann['image_id']]
        img_w, img_h = img_info['width'], img_info['height']
        cat_id = categories[ann['category_id']]
        x, y, w, h = ann['bbox']
        xc = (x + w / 2) / img_w
        yc = (y + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h

        label_path = yolo_labels / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_path, "a") as f:
            f.write(f"{cat_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
coco_to_yolo("task2_annotation_coco.json", "labels", "task2")
