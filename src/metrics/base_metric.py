import torch
from enum import Enum
from src.metrics.cosine_similarity import CosineSimilarity
from src.metrics.euclidian import EuclideanDistance
from src.metrics.mahalanobis import MahalanobisDistance
from src.metrics.mahalanobis_kalman import MahalanobisKalmanDistance

class MetricType(Enum):
    COSINE_SIMILARITY = 1
    EUCLIDEAN_DISTANCE = 2
    MAHALANOBIS_DISTANCE = 3
    MAHALANOBIS_KALMAN_DISTANCE = 4

class BaseMetric:
    def __init__(self, metric: MetricType = MetricType.COSINE_SIMILARITY ) -> None:
        self.metric_type = metric
        if metric == MetricType.COSINE_SIMILARITY:
            self._metric = CosineSimilarity()
        elif metric == MetricType.EUCLIDEAN_DISTANCE:
            self._metric = EuclideanDistance()
        elif metric == MetricType.MAHALANOBIS_DISTANCE:
            self._metric = MahalanobisDistance()
        elif metric == MetricType.MAHALANOBIS_KALMAN_DISTANCE:
            self._metric = MahalanobisKalmanDistance()
        else:
            raise NotImplementedError("Metric not implemented")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor, kalman_distances: torch.Tensor | None = None) -> tuple[list[int], list[float]]:
        if kalman_distances is not None: 
            assert self.metric_type == MetricType.MAHALANOBIS_KALMAN_DISTANCE
            return self._metric.rank(target_features, query_features, kalman_distances) # type: ignore
        return self._metric.rank(target_features, query_features)

