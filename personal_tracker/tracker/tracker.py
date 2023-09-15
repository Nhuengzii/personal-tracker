from typing import Any
from personal_tracker.metric.metric import Metric
from personal_tracker.metric.metric_type import MetricType
from .track_result import TrackResult
from personal_tracker.kalman_filter import KalmanFilter
from personal_tracker.detector import Detector
from personal_tracker.embedder import Embedder
from personal_tracker.helpers import draw_bbox, rec_check
from datetime import datetime, timedelta
from cv2.typing import MatLike
import numpy as np
import torch
import cv2
from collections import deque


class Tracker():
    def __init__(self, detector: Detector, embedder: Embedder | None, metric_type: MetricType, auto_add_target_features: bool = False, auto_add_target_features_interval: int = 60,
                 sift_history_size: int = 10, scores_history_size: int = 10) -> None:
        self._detector = detector
        self._embedder = embedder
        self._metric = Metric(metric_type)
        self.kf = KalmanFilter()
        self._target_features_pool: deque[tuple[MatLike, torch.Tensor | None, Any]] = deque(maxlen=200)
        self._last_update_time = datetime.now()
        self._last_auto_add_target_features_time = datetime.now()
        self.auto_add_target_features = auto_add_target_features
        self.auto_add_target_features_interval = auto_add_target_features_interval
        self._last_track_result: TrackResult | None = None
        self._k_missed = 0 # kalman missed (algorithm success but not the same as kalman)
        self._t_missed = 0 # track missed (target not in frame)
        self._k_missed_threshold = 3 # kalman missed threshold
        self._t_missed_threshold = 30 # track missed threshold
        self.sift = cv2.SIFT_create() # type: ignore
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self._sift_history: deque[Any] = deque(maxlen=sift_history_size)
        self._scores_history: deque[float] = deque(maxlen=scores_history_size)

    def track(self, frame: MatLike, draw_result: bool = False) -> TrackResult:
        raw_result = self._algorithm(frame)
        result = self._kalman_ensuring(frame, raw_result)
        if result is not None and result.success:
            if draw_result: self.draw_result(frame, result);
            self._scores_history.append(result.target_score)
        if self._should_add_target_features(result):
            assert result.target_bbox is not None
            self.add_target_features(frame, result.target_bbox)
            self._last_auto_add_target_features_time = datetime.now()
        self._last_update_time = datetime.now()
        self._last_track_result = result
        return result

    def _algorithm(self, frame: MatLike) -> TrackResult:
        detection_result = self._detector.detect(frame)
        if detection_result is None:
            return TrackResult()
        embedder_scores: list[float] | None = None
        if self._embedder is not None:
            detected_features = self._embedder.extract_features(frame, detection_result.bboxes)
            target_features: list[torch.Tensor] = []
            for target in self._target_features_pool:
                features = target[1]
                target_features.append(features.tolist()) # type: ignore
            embedder_ranks, sorted_embedder_scores = self._metric.rank(torch.tensor(target_features), detected_features)
            embedder_scores = self._unsorted_scores(embedder_ranks, sorted_embedder_scores)
            
        # Sift ranking 
        q_keypoints = []
        for idx in range(len(detection_result.bboxes)):
            croped = detection_result.get_crop_and_remove_background(idx, True)
            q_keypoints.append(self._get_sift_keypoints(croped))
        matches = [0 for _ in range(len(q_keypoints))]
        for idx in range(len(self._target_features_pool)):
            try:
                ms = self._get_matches_sift_keypoints(self._target_features_pool[idx][2], q_keypoints)
            except:
                return TrackResult(detection_result)
            assert len(ms) == len(matches)
            for i in range(len(ms)):
                matches[i] += ms[i]
        # Sift bias from history
        for s in self._sift_history:
            try:
                ms = self._get_matches_sift_keypoints(s, q_keypoints)
            except:
                return TrackResult(detection_result)
            assert len(ms) == len(matches)
            for i in range(len(ms)):
                matches[i] += ms[i]

        if embedder_scores is not None:
            assert embedder_scores is not None
            assert len(matches) == len(embedder_scores)
            if self._metric.metric_type in [MetricType.CSEM_DISTANCE, MetricType.MAHALANOBIS_DISTANCE, MetricType.EUCLIDEAN_DISTANCE]:
                # inverse the distance
                mean_embedder_scores = sum(embedder_scores) / len(embedder_scores)
                for idx in range(len(embedder_scores)):
                    embedder_scores[idx] = mean_embedder_scores / embedder_scores[idx]
            combined_scores = [0.0 for i in range(len(matches))]
            for idx in range(len(matches)):
                combined_scores[idx] = 2 * matches[idx] * embedder_scores[idx] / (matches[idx] + embedder_scores[idx])
            sorted_scores, ranks = torch.sort(torch.tensor(combined_scores), descending=True) # from high to low
        else:
            sorted_scores, ranks = torch.sort(torch.tensor(matches), descending=True) # from high to low
        
        assert isinstance(ranks, torch.Tensor)
        assert isinstance(sorted_scores, torch.Tensor)
        target_idx = int(ranks[0].item())

        metric_type = self._metric.metric_type
        if metric_type in [MetricType.CSEM_DISTANCE, MetricType.MAHALANOBIS_DISTANCE, MetricType.EUCLIDEAN_DISTANCE]:
            percent = 0.90
        else:
            percent = 0.95
        if len(self._scores_history) > 0:
            mean_score_history = sum(self._scores_history) / len(self._scores_history)
            if sorted_scores[0].item() < mean_score_history * percent:
                # print(f"Score too low. skipping ... {sorted_scores[0].item()} < {mean_score_history * percent}")
                return TrackResult(detection_result)
        result = TrackResult(detection_result, target_idx, ranks.tolist(), sorted_scores.tolist())
        return result

    def _unsorted_scores(self, ranks: list[int], sorted_scores: list[float]) -> list[float]:
        unsorted_scores = [float(i) for i in range(len(ranks))]
        for idx, rank in enumerate(ranks):
            unsorted_scores[rank] = sorted_scores[idx]
        return unsorted_scores

    def _get_sift_keypoints(self, img: MatLike):
        return self.sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    
    def _get_matches_sift_keypoints(self, target_keypoint, query_keypoints) -> list[int]:
        num_matches = []
        for q in query_keypoints:
            matches = self.bf.match(target_keypoint[1], q[1])
            matches = sorted(matches, key=lambda x: x.distance)
            num_matches.append(len(matches))
        return num_matches

    def _kalman_ensuring(self, frame: MatLike, raw_result: TrackResult) -> TrackResult:
        if not raw_result.success:
            self._t_missed += 1
            self._last_track_result = raw_result
            return raw_result
        assert raw_result.target_idx is not None
        if self._last_track_result is None or not self._last_track_result.success:
            return raw_result
        algorithm_target_idx = raw_result.target_idx     
        last_target_bbox = self._last_track_result.target_bbox
        assert last_target_bbox is not None
        cx, cy = (last_target_bbox[0] + last_target_bbox[2]) // 2, (last_target_bbox[1] + last_target_bbox[3]) // 2
        pcx, pcy = self.kf.predict(cx, cy)
        assert raw_result.detect_result is not None
        kal_rank = self._kalman_ranking((pcx, pcy), raw_result.detect_result.bboxes)
        if kal_rank[0] == algorithm_target_idx:
            self._k_missed = 0
            return raw_result
        self._k_missed += 1
        if self._k_missed > self._k_missed_threshold:
            self._k_missed = 0
            return raw_result

        # Kalman overwrite
        raw_result.k_overwrite(raw_result.detect_result.bboxes[kal_rank[0]])
        return raw_result

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
        detect_result = self._detector.detect(croped_image)
        if detect_result is None:
            print("Can't detect any object. skipping ...")
            return
        # assert detect_result.num_detected == 1, "detect_result.num_detected should be 1"
        rm_bg = detect_result.get_crop_and_remove_background(0)
        kp, des = self._get_sift_keypoints(rm_bg)
        if self._embedder is None:
            self._target_features_pool.append((croped_image, None, (kp, des)))
            return
        features = self._embedder.extract_feature(croped_image)
        self._target_features_pool.append((croped_image, features, (kp, des)))
        
    def draw_result(self, frame: MatLike, track_result: TrackResult) -> None:
        assert track_result.success
        target_bbox = track_result.target_bbox
        assert target_bbox is not None
        c = (0, 255, 0)
        if track_result.is_overwrited:
            c = (255, 0, 255)
        draw_bbox(frame, target_bbox, c, str(track_result.sorted_scores[track_result.target_idx])) # type: ignore
    
    def _should_add_target_features(self, track_result: TrackResult) -> bool:
        if not self.auto_add_target_features:
            return False
        if (datetime.now() - self._last_auto_add_target_features_time).seconds < self.auto_add_target_features_interval:
            # # penalty last_time by subtracting 1 / 4 of interval
            # self._last_auto_add_target_features_time -= timedelta(seconds=int(self.auto_add_target_features_interval / 4))
            return False
        if track_result.detect_result is None:
            return False
        bboxes = track_result.detect_result.bboxes 
        target_idx = track_result.target_idx
        if target_idx is None:
            return False
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
