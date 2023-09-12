from cv2.typing import MatLike
from datetime import datetime
import torch
from src.detectors.base_detector import BaseDetector
from src.embedders.base_embedder import BaseEmbedder
from src.kalman_filter import KalmanFilter
from src.metrics.base_metric import BaseMetric, MetricType
from src.metrics.me_metric import MEMetric
from src.personal_trackers.personal_tracker import PersonalTracker
from src.personal_trackers.track_result import TrackResults


class MEPersonalTracker(PersonalTracker):
    def __init__(self, detector: BaseDetector, embedders: list[BaseEmbedder], metric: MetricType, auto_add_target_features: bool = False, auto_add_target_features_interval: int = 60) -> None:
        assert metric != MetricType.MAHALANOBIS_KALMAN_DISTANCE, "ME tracker does not support Mahalanobis Kalman Distance"
        self.metric = BaseMetric(metric)
        self._metric_type = metric
        self.kf = KalmanFilter()
        self.auto_add_target_features = auto_add_target_features
        self.auto_add_target_features_interval = auto_add_target_features_interval
        self.detector = detector
        self.embedders = embedders
        self.auto_add_target_features = auto_add_target_features
        self.auto_add_target_features_interval = auto_add_target_features_interval
        self._target_features_pool: list[tuple[MatLike, list[torch.Tensor]]] = []
        self._last_track_result: TrackResults | None = None
        self._last_update_time = datetime.now()
        self.trust_kalman = True
        self._argued_missed_cont = 0
        self._kalman_trust_threshold = 3

    def _algorithm(self, frame: MatLike) -> TrackResults | None:
        detect_result = self.detector.detect(frame)
        if detect_result is None:
            return None
        me_detected_features = []
        me_target_features = []
        for idx, embedder in enumerate(self.embedders):
            me_detected_features.append(embedder.extract_features(frame, detect_result.bboxes))
            target_features = []
            for i in range(len(self._target_features_pool)):
                target_features.append(self._target_features_pool[i][1][idx])
            me_target_features.append(target_features)

        normalized_scores: list[list[float]] = []
        assert len(me_detected_features) == len(me_target_features)
        for i in range(len(me_target_features)):
            ranks, sorted_scores = self.metric.rank(me_target_features[i], me_detected_features[i])
            if len(ranks) == 1:
                normalized_scores.append(sorted_scores)
                continue
            unsorted_scores = self._unsorted_scores(ranks, sorted_scores)
            print(f"unsorted_scores: {unsorted_scores}")
            mean_score = sum(unsorted_scores) / len(unsorted_scores)
            std_score = sum([(score - mean_score) ** 2 for score in unsorted_scores]) / len(unsorted_scores)
            print(f"mean_score: {mean_score}, std_score: {std_score}")
            assert std_score != 0
            normalized_score = [(score - mean_score) / std_score for score in unsorted_scores]
            normalized_scores.append(normalized_score)

        sum_normalized_scored: list[float] = []
        for i in range(len(normalized_scores[0])):
            sum_normalized_scored.append(sum([score[i] for score in normalized_scores]))
        if self._metric_type == MetricType.COSINE_SIMILARITY:
            sorted_scores, ranks = torch.sort(torch.tensor(sum_normalized_scored), descending=True)
        else:
            sorted_scores, ranks = torch.sort(torch.tensor(sum_normalized_scored), descending=False)
        ranks: list[int] = ranks.tolist() # type: ignore
        sorted_scores: list[float] = sorted_scores.tolist() # type: ignore
        target_idx = ranks[0]
        return TrackResults(detect_result, target_idx, ranks, sorted_scores)
        
        

    def _unsorted_scores(self, ranks: list[int], sorted_scores: list[float] ):
        unsorted_scores = [float(i) for i in range(len(ranks))]
        for idx, rank in enumerate(ranks):
            unsorted_scores[rank] = sorted_scores[idx]
        return unsorted_scores
    
    def add_target_features(self, cv_image: MatLike, bbox: tuple[int, int, int, int]) -> None:
        croped_image = cv_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        features = []
        for embedder in self.embedders:
            features.append(embedder.extract_feature(croped_image))
        self._target_features_pool.append((croped_image, features))
