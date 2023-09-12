#https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
import cv2
from datetime import datetime
from cv2.typing import MatLike
import numpy as np
from src.detectors.base_detector import BaseDetector
from src.helpers import draw_bbox
from src.embedders.base_embedder import BaseEmbedder
from src.metrics.base_metric import MetricType
from src.personal_trackers.base_personal_tracker import BasePersonalTracker, TrackResults
import torch


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

class KalmanPersonalTracker(BasePersonalTracker):
    def __init__(self, detector: BaseDetector, embedder: BaseEmbedder, metric: MetricType, auto_add_target_features: bool = False, auto_add_target_features_interval: int = 60) -> None:
        super().__init__(detector, embedder, metric, auto_add_target_features, auto_add_target_features_interval)
        self.kf = KalmanFilter()
        self._argued_missed_cont = 0
        self.trust_kalman = True

    def track(self, frame: MatLike, draw_result: bool = False) -> TrackResults | None:
        detect_result = self.detector.detect(frame)
        if detect_result is None:
            return None
        detected_features = self.embedder.extract_features(frame, detect_result.bboxes)
        target_features: list[torch.Tensor] = []
        for _, features in self._target_features_pool:
            target_features.append(features.tolist()) # type: ignore
        ranks, sorted_scores = self.metric.rank(torch.tensor(target_features), detected_features)
        target_idx = ranks[0]
        cx, cy = None, None
        pcx, pcy = None, None
        if self._last_track_result is not None:
            last_target_bbox = self._last_track_result.detect_result.bboxes[self._last_track_result.target_idx]
            cx, cy = (last_target_bbox[0] + last_target_bbox[2]) // 2, (last_target_bbox[1] + last_target_bbox[3]) // 2
            pcx, pcy = self.kf.predict(cx, cy)
            if self.metric.metric_type == MetricType.MAHALANOBIS_KALMAN_DISTANCE:
                kalman_distances = self._kalman_distances((pcx, pcy), detect_result.bboxes)
                ranks, sorted_scores = self.metric.rank(torch.tensor(target_features), detected_features, kalman_distances)
                target_idx = ranks[0]
            kal_rank = self._kalman_ranking((pcx, pcy), detect_result.bboxes)
            if kal_rank[0] != target_idx:
                self._argued_missed_cont += 1
                if self.trust_kalman and self._argued_missed_cont <= 3:
                    target_idx = kal_rank[0]
                if self.trust_kalman and self._argued_missed_cont > 3:
                    self._argued_missed_cont = 0
                    self.trust_kalman = False
            else:
                self._argued_missed_cont = 0
                self.trust_kalman = True

        if draw_result:
            self.draw_result(frame, TrackResults(detect_result, target_idx, ranks, sorted_scores), (cx, cy), (pcx, pcy))
        if self.auto_add_target_features and self._last_auto_add_target_features_time and (datetime.now() - self._last_auto_add_target_features_time).total_seconds() > self.auto_add_target_features_interval:
            if self._should_add_target_features(TrackResults(detect_result, target_idx, ranks, sorted_scores)):
                self.add_target_features(frame, detect_result.bboxes[target_idx])
                self._last_auto_add_target_features_time = datetime.now()
        self._last_update_time = datetime.now()
        self._last_track_result = TrackResults(detect_result, target_idx, ranks, sorted_scores)
        return TrackResults(detect_result, target_idx, ranks, sorted_scores)
    
    def _kalman_ranking(self, kalman_predict: tuple[int, int], bboxes: list[tuple[int, int, int, int]]) -> list[int]:
        distances = []
        pcx, pcy = kalman_predict
        for bbox in bboxes:
            distances.append(self._kalman_distance(kalman_predict, bbox))
        return np.argsort(distances).tolist()
    def _kalman_distances(self, kalman_predict: tuple[int, int], bboxes: list[tuple[int, int, int, int]]) -> torch.Tensor:
        distances = []
        pcx, pcy = kalman_predict
        for bbox in bboxes:
            distances.append(self._kalman_distance(kalman_predict, bbox))
        return torch.tensor(distances) # tensor of shape (num_query_samples, )
    def _kalman_distance(self, kalman_predict: tuple[int, int], bbox: tuple[int, int, int, int]) -> float:
        pcx, pcy = kalman_predict
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        dis = (pcx - cx) ** 2 + (pcy - cy) ** 2
        return dis ** 0.5
        
    def draw_result(self, frame: MatLike, track_result: TrackResults, cxy: tuple[int | None, int | None] , pcxy: tuple[int | None, int | None] ) -> None:
        detect_result = track_result.detect_result
        target_idx = track_result.target_idx

        for idx, bbox in enumerate(detect_result.bboxes):
            if idx == target_idx:
                draw_bbox(frame, bbox, (0, 255, 0), f"Target {track_result.sorted_scores[target_idx]}") 
            else:
                draw_bbox(frame, bbox, (0, 0, 255))
        if self.trust_kalman:
            cv2.putText(frame, "Kalman", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Metric", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cx, cy = cxy
        pcx, pcy = pcxy
        if cx is not None and cy is not None and pcx is not None and pcy is not None:
            cv2.circle(frame, (cx, cy), 20, (255, 0, 0), -1)
            cv2.putText(frame, "C", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(frame, (pcx, pcy), 20, (0, 255, 0), -1)
            cv2.putText(frame, "P", (pcx, pcy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
         
