import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results
from cv2.typing import MatLike


class DetectorResult:
    def __init__(self, result_boxes: Boxes) -> None:
        self.bboxes = result_boxes.xyxy.tolist()
        self.bboxes: list[tuple[int, int, int, int]] = [tuple(map(int, bbox)) for bbox in self.bboxes]
        self.confidences = result_boxes.conf.tolist()
        self.classes = result_boxes.cls.tolist()
        self.num_boxes = len(self.bboxes)
    def __len__(self) -> int:
        return self.num_boxes
    def __getitem__(self, index: int) -> tuple[tuple[int, int, int, int], float, int]:
        return self.bboxes[index], self.confidences[index], self.classes[index]

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
        return DetectorResult(result.boxes)
        
        

    
        
