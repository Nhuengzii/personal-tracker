from src.detectors.base_detector import BaseDetector
from ultralytics import YOLO


class KeypointsDetector(BaseDetector):
    def __init__(self) -> None:
        super().__init__()
        self.model = YOLO('yolov8n-pose.pt')
