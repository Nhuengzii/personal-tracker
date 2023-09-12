import cv2
import torch
from ultralytics import YOLO
from src.detectors.detector_result import DetectorResult
from ultralytics.engine.results import Boxes, Results
from cv2.typing import MatLike

class BaseDetector:
    def __init__(self) -> None:
        self.model = YOLO('yolov8n.pt') 

    def detect(self, cv_image: MatLike ) -> DetectorResult | None:
        results: list[Results] = self.model.predict(cv_image, verbose=False, classes=[0])
        if len(results) == 0:
            return None
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            print("pass loop because result.boxes is None or len(result.boxes) == 0")
            return None
        return DetectorResult(result)
        
        

    
        
