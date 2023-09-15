from ultralytics import YOLO
from ultralytics.engine.results import Results
from cv2.typing import MatLike

from personal_tracker.detector.detector_result import DetectorResult

class Detector:
    def __init__(self) -> None:
        self._model = YOLO('yolov8n-seg.pt') 

    def detect(self, cv_image: MatLike ) -> DetectorResult | None:
        results: list[Results] = self._model.predict(cv_image, verbose=False, classes=[0])
        if len(results) == 0:
            return None
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        return DetectorResult(result)
