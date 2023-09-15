from cv2.typing import MatLike
import cv2
from datetime import datetime
from personal_tracker.detector.detector import Detector
from personal_tracker.embedder.emebedder import Embedder
from personal_tracker.metric import Metric
from personal_tracker.metric.metric_type import MetricType
from personal_tracker.tracker.track_result import TrackResult
from personal_tracker.tracker.tracker import Tracker


class PersonalTracker:
    def __init__(self) -> None:
        self._detector = Detector()
        self._embedder = Embedder()
        self.metric_type = MetricType.COSINE_SIMILARITY
        self._tracker = Tracker(self._detector, self._embedder, self.metric_type)

    def track(self, frame: MatLike, draw_result: bool = False) -> TrackResult:
        result = self._tracker.track(frame, draw_result)
        return result

    def add_target_features(self, frame: MatLike, soi: tuple[int, int, int, int]) -> None:
        self._tracker.add_target_features(frame, soi)

    def get_target_from_camera(self, cap: cv2.VideoCapture, num_target: int = 1) -> list[tuple[MatLike, tuple[int, int, int, int]]]:
        targets = []
        _count = 0
        while _count < num_target:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). continue ...")
                continue
            cv2.imshow("target", frame)
            if cv2.waitKey(20) & 0xFF == ord('s'):
                target_frame = frame
                soi = cv2.selectROI("target", target_frame) 
                # convert to x1, y1, x2, y2
                soi = (int(soi[0]), int(soi[1]), int(soi[0]) + int(soi[2]), int(soi[1]) + int(soi[3]))
                targets.append((target_frame, soi))
                _count += 1
        cv2.destroyAllWindows()
        return targets

    def get_repetitive_target_from_camera(self, cap: cv2.VideoCapture, sec: int) -> list[tuple[MatLike, tuple[int, int, int, int]]]:
        targets = []
        start = datetime.now()
        while True:
            if (datetime.now() - start).total_seconds() > sec:
                break
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). continue ...")
                continue
            cv2.imshow("target", frame)
            target_frame = frame
            detected = self._detector.detect(target_frame)
            if not detected:
                print("Can't detect any object. continue ...")
                continue
            soi = detected.bboxes[0]
            # convert to x1, y1, x2, y2
            targets.append((target_frame, soi))
            cv2.waitKey(25)
        cv2.destroyAllWindows()
        return targets

