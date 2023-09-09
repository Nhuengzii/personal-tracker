from datetime import datetime
from cv2.typing import MatLike
import torch
from src.detectors import BaseDetector, DetectorResult
from src.embedders import BaseEmbedder
from src.helpers import draw_bbox, rec_check
import cv2

from src.metrics.base_metric import BaseMetric, MetricType

class TrackResults:
    def __init__(self, detect_result: DetectorResult, target_idx: int) -> None:
        self.detect_result = detect_result
        self.target_idx = target_idx

class BasePersonalTracker():
    def __init__(self, detector: BaseDetector, embedder: BaseEmbedder, metric: MetricType, auto_add_target_features: bool = False, auto_add_target_features_interval: int = 60) -> None:
        self.detector = detector
        self.embedder = embedder
        self.metric = BaseMetric(metric)
        self._target_features_pool: list[tuple[MatLike, torch.Tensor]] = []
        self._last_update_time = datetime.now()
        self._last_auto_add_target_features_time = datetime.now()
        self.auto_add_target_features = auto_add_target_features
        self.auto_add_target_features_interval = auto_add_target_features_interval

    def track(self, frame: MatLike, draw_result: bool = False) -> TrackResults | None:
        detect_result = self.detector.detect(frame)
        if detect_result is None:
            return None
        detected_features = self.embedder.extract_features(frame, detect_result.bboxes)
        target_features: list[torch.Tensor] = []
        for _, features in self._target_features_pool:
            target_features.append(features.tolist())
        ranks = self.metric.rank(torch.tensor(target_features), detected_features)
        target_idx = ranks[0]
        if draw_result:
            self.draw_result(frame, TrackResults(detect_result, target_idx))
        if self.auto_add_target_features and self._last_auto_add_target_features_time and (datetime.now() - self._last_auto_add_target_features_time).total_seconds() > self.auto_add_target_features_interval:
            if self._should_add_target_features(TrackResults(detect_result, target_idx)):
                self.add_target_features(frame, detect_result.bboxes[target_idx])
                self._last_auto_add_target_features_time = datetime.now()
        self._last_update_time = datetime.now()
        return TrackResults(detect_result, target_idx)
    
    def add_target_features(self, cv_image: MatLike, bbox: tuple[int, int, int, int]) -> None:
        croped_image = cv_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        features = self.embedder.extract_feature(croped_image)
        self._target_features_pool.append((croped_image, features))
        
    def draw_result(self, frame: MatLike, track_result: TrackResults) -> None:
        detect_result = track_result.detect_result
        target_idx = track_result.target_idx

        for idx, bbox in enumerate(detect_result.bboxes):
            if idx == target_idx:
                draw_bbox(frame, bbox, (0, 255, 0), "Target") 
            else:
                draw_bbox(frame, bbox, (0, 0, 255))
    
    def _should_add_target_features(self, track_result: TrackResults) -> bool:
        bboxes = track_result.detect_result.bboxes 
        target_idx = track_result.target_idx
        if target_idx == -1:
            return False
        check = rec_check(bboxes, target_idx, 20)
        if check:
            return True
        return False
    def show_target_images(self) -> MatLike:
        target_images = []
        for target_image, _ in self._target_features_pool:
            target_images.append(cv2.resize(target_image, (128, 256)))

        # concat images in horizontal
        target_images = cv2.hconcat(target_images)
        return target_images
        
