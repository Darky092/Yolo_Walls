
from ultralytics import YOLO

def main():
    model = YOLO('runs/segment/train8/weights/best.pt')
    results = model.train(data="dataset.yaml", epochs=50, imgsz=640, batch=32)

if __name__ == '__main__':
    main()