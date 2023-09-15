from cv2.typing import MatLike
import cv2
from datetime import datetime
from personal_tracker.detector.detector import Detector
from personal_tracker.embedder.available_embedder_models import AvailableEmbedderModels
from personal_tracker.embedder.clip_embedder import CLIPEmbedder
from personal_tracker.embedder.emebedder import Embedder
from personal_tracker.metric import Metric
from personal_tracker.metric.metric_type import MetricType
from personal_tracker.tracker import Tracker, TrackResult, TrackerConfig


class PersonalTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self._detector = Detector()
        if config.embedder_model is None:
            self._embedder = None
        elif config.embedder_model == AvailableEmbedderModels.CLIP:
            self._embedder = CLIPEmbedder()
        else:
            self._embedder = Embedder(config.embedder_model)
        self.metric_type = config.metric_type
        self._tracker = Tracker(self._detector,
                                self._embedder,
                                self.metric_type,
                                sift_history_size=config.sift_history_size,
                                auto_add_target_features=config.auto_add_target_features,
                                auto_add_target_features_interval=config.auto_add_target_features_interval)

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
            cv2.waitKey(100)
        cv2.destroyAllWindows()
        return targets

