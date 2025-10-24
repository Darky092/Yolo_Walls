from ultralytics import YOLO
import torch
import os 

def Training():
    model = YOLO("yolo11n.yaml")
    results = model.train(data="Dataset.yaml", epochs=10)
