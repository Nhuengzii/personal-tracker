from .track_result import TrackResults
from src.kalman_filter import KalmanFilter
from src.detectors import BaseDetector
from src.embedders import BaseEmbedder
from src.helpers import draw_bbox, rec_check
from src.metrics.base_metric import BaseMetric, MetricType
from datetime import datetime
from cv2.typing import MatLike
import numpy as np
import torch
import cv2


class PersonalTracker():
    def __init__(self, detector: BaseDetector, embedder: BaseEmbedder, metric: MetricType, auto_add_target_features: bool = False, auto_add_target_features_interval: int = 60) -> None:
        self.detector = detector
        self.embedder = embedder
        self.metric = BaseMetric(metric)
        self.kf = KalmanFilter()
        self._target_features_pool: list[tuple[MatLike, torch.Tensor]] = []
        self._last_update_time = datetime.now()
        self._last_auto_add_target_features_time = datetime.now()
        self.auto_add_target_features = auto_add_target_features
        self.auto_add_target_features_interval = auto_add_target_features_interval
        self._last_track_result: TrackResults | None = None
        self.trust_kalman = True
        self._argued_missed_cont = 0
        self._kalman_trust_threshold = 3

    def track(self, frame: MatLike, draw_result: bool = False) -> TrackResults | None:
            
        raw_result = self._algorithm(frame)
        if raw_result is None:
            return None
        result = self._kalman_ensuring(frame, raw_result)
        
        if result is None:
            return None
        if draw_result:
            self.draw_result(frame, result)
        self._last_update_time = datetime.now()
        self._last_track_result = result

    def _algorithm(self, frame: MatLike) -> TrackResults | None:
        detect_result = self.detector.detect(frame)
        if detect_result is None:
            return None
        detected_features = self.embedder.extract_features(frame, detect_result.bboxes)
        target_features: list[torch.Tensor] = []
        for target in self._target_features_pool:
            features = target[1]
            target_features.append(features.tolist()) # type: ignore
        ranks, sorted_scores = self.metric.rank(torch.tensor(target_features), detected_features)
        target_idx = ranks[0]
        return TrackResults(detect_result, target_idx, ranks, sorted_scores, detected_features, detected_features)
    def _kalman_ensuring(self, frame: MatLike, raw_result: TrackResults) -> TrackResults | None:
        if raw_result is None:
            return None
        cx, cy = None, None
        pcx, pcy = None, None
        detect_result = raw_result.detect_result
        target_features = raw_result.target_features
        detected_features = raw_result.detected_features
        target_idx = raw_result.target_idx     
        ranks = raw_result.ranks
        sorted_scores = raw_result.sorted_scores
        if self._last_track_result is not None:
            last_target_bbox = self._last_track_result.target_bbox
            cx, cy = (last_target_bbox[0] + last_target_bbox[2]) // 2, (last_target_bbox[1] + last_target_bbox[3]) // 2
            pcx, pcy = self.kf.predict(cx, cy)
            if self.metric.metric_type == MetricType.MAHALANOBIS_KALMAN_DISTANCE:
                kalman_distances = self._kalman_distances((pcx, pcy), detect_result.bboxes)
                assert detected_features is not None
                ranks, sorted_scores = self.metric.rank(torch.tensor(target_features), detected_features, kalman_distances)
                target_idx = ranks[0]
            kal_rank = self._kalman_ranking((pcx, pcy), detect_result.bboxes)
            if kal_rank[0] != target_idx:
                self._argued_missed_cont += 1
                if self.trust_kalman and self._argued_missed_cont <= self._kalman_trust_threshold:
                    target_idx = kal_rank[0]
                if self.trust_kalman and self._argued_missed_cont > self._kalman_trust_threshold:
                    self._argued_missed_cont = 0
                    self.trust_kalman = False
            else:
                self._argued_missed_cont = 0
                self.trust_kalman = True
        result = TrackResults(detect_result, target_idx, ranks, sorted_scores, target_features, detected_features)
        return result
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
    
    def add_target_features(self, cv_image: MatLike, bbox: tuple[int, int, int, int]) -> None:
        croped_image = cv_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        features = self.embedder.extract_feature(croped_image)
        self._target_features_pool.append((croped_image, features))
        
    def draw_result(self, frame: MatLike, track_result: TrackResults) -> None:
        detect_result = track_result.detect_result
        target_idx = track_result.target_idx

        for idx, bbox in enumerate(detect_result.bboxes):
            if idx == target_idx:
                draw_bbox(frame, bbox, (0, 255, 0), f"Target: {track_result.sorted_scores[idx]}") 
            else:
                draw_bbox(frame, bbox, (0, 0, 255), f"{track_result.sorted_scores[idx]}")
    
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
        for target in self._target_features_pool:
            target_images.append(cv2.resize(target[0], (128, 256)))
        # concat images in horizontal
        target_images = cv2.hconcat(target_images)
        return target_images
    
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
            detected = self.detector.detect(target_frame)
            if not detected:
                print("Can't detect any object. continue ...")
                continue
            soi = detected.bboxes[0]
            # convert to x1, y1, x2, y2
            targets.append((target_frame, soi))
            cv2.waitKey(25)
        cv2.destroyAllWindows()
        return targets
