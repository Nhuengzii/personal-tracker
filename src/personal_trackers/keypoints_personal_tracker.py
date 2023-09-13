import torch
import cv2
from datetime import datetime
from cv2.typing import MatLike
from src.detectors.base_detector import BaseDetector
from src.detectors.detector_result import DetectorResult
from src.detectors.keypoints_detector import KeypointsDetector
from src.embedders.base_embedder import BaseEmbedder
from src.helpers import draw_bbox
from src.metrics.base_metric import MetricType
from src.personal_trackers.personal_tracker import PersonalTracker
from src.personal_trackers.track_result import TrackResults


class KeypointsPersonalTracker(PersonalTracker):
    def __init__(self, detector: KeypointsDetector, embedder: BaseEmbedder, metric: MetricType,
                 auto_add_target_features: bool = False, auto_add_target_features_interval: int = 60) -> None:
        super().__init__(detector, embedder, metric, auto_add_target_features, auto_add_target_features_interval)
        self._target_features_pool_direction: list[str] = []

    def add_target_features(self, cv_image: MatLike, bbox: tuple[int, int, int, int]) -> None:
        croped_image = cv_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        detect_result = self.detector.detect(croped_image)
        if detect_result is None:
            raise Exception("Can't detect keypoints")
        assert len(detect_result) == 1, "target image must have only one person"
        features = self.embedder.extract_feature(croped_image)
        direction = detect_result.direction(0)
        self._target_features_pool_direction.append(direction)
        self._target_features_pool.append((croped_image, features))

    def _query_target_features(self, direction: str) -> list[torch.Tensor]:
        assert len(self._target_features_pool) >= 3 # FRONT, BACK, SIDE
        assert direction in ["FRONT", "BACK", "SIDE"], "direction must be FRONT, BACK or SIDE"
        target_features: list[torch.Tensor] = []
        for idx, (direction, features) in enumerate(zip(self._target_features_pool_direction, self._target_features_pool)):
            if direction == direction:
                target_features.append(features[1])
        return target_features

    def _rank(self, detected_result: DetectorResult, detected_features: torch.Tensor) -> tuple[list[int], list[float]]:
        unsorted_score = []
        for i in range(len(detected_result)):
            direction = detected_result.direction(i)
            target_features = self._query_target_features(direction)
            _, scores = self.metric.rank(torch.cat(target_features), detected_features[i].unsqueeze(0))
            unsorted_score.append(scores[0])
        assert len(unsorted_score) == len(detected_result)
        if self.metric.metric_type == MetricType.COSINE_SIMILARITY:
            sorted_scores, ranks = torch.sort(torch.tensor(unsorted_score), descending=True)
        else:
            sorted_scores, ranks = torch.sort(torch.tensor(unsorted_score), descending=False)
        return ranks.tolist(), sorted_scores.tolist()

    def draw_result(self, frame: MatLike, track_result: TrackResults) -> None:
        detect_result = track_result.detect_result
        target_idx = track_result.target_idx

        for idx, bbox in enumerate(detect_result.bboxes):
            if idx == target_idx:
                draw_bbox(frame, bbox, (0, 255, 0), f"Target: {detect_result.direction(idx)}") 
            else:
                draw_bbox(frame, bbox, (0, 0, 255), f"{detect_result.direction(idx)}")

    def track(self, frame: MatLike, draw_result: bool = False) -> TrackResults | None:
        detect_result = self.detector.detect(frame)
        if detect_result is None:
            return None
        detected_features = self.embedder.extract_features(frame, detect_result.bboxes)
        ranks, sorted_scores = self._rank(detect_result, detected_features)
        target_idx = ranks[0]
        if draw_result:
            self.draw_result(frame, TrackResults(detect_result, target_idx, ranks, sorted_scores))
        if self.auto_add_target_features and self._last_auto_add_target_features_time and (datetime.now() - self._last_auto_add_target_features_time).total_seconds() > self.auto_add_target_features_interval:
            if self._should_add_target_features(TrackResults(detect_result, target_idx, ranks, sorted_scores)):
                self.add_target_features(frame, detect_result.bboxes[target_idx])
                self._last_auto_add_target_features_time = datetime.now()
        self._last_update_time = datetime.now()
        self._last_track_result = TrackResults(detect_result, target_idx, ranks, sorted_scores)
        return TrackResults(detect_result, target_idx, ranks, sorted_scores)

    def show_target_images(self) -> MatLike:
        target_images = []
        for idx, (target_image, _) in enumerate(self._target_features_pool):
            direction = self._target_features_pool_direction[idx]
            target_image = cv2.putText(target_image, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            target_images.append(cv2.resize(target_image, (128, 256)))
        # concat images in horizontal
        target_images = cv2.hconcat(target_images)
        return target_images

