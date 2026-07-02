# src/detector.py

from ultralytics import YOLO


class TomatoDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, frame, conf: float = 0.4):
        results = self.model(frame, conf=conf)
        return results[0].plot()