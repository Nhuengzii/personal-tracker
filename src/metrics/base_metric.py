import torch
from enum import Enum
from src.metrics.cosine_similarity import CosineSimilarity
from src.metrics.euclidian import EuclideanDistance
from src.metrics.mahalanobis import MahalanobisDistance

class MetricType(Enum):
    COSINE_SIMILARITY = 1
    EUCLIDEAN_DISTANCE = 2
    MAHALANOBIS_DISTANCE = 3

class BaseMetric:
    def __init__(self, metric: MetricType = MetricType.COSINE_SIMILARITY ) -> None:
        if metric == MetricType.COSINE_SIMILARITY:
            self._metric = CosineSimilarity()
        elif metric == MetricType.EUCLIDEAN_DISTANCE:
            self._metric = EuclideanDistance()
        elif metric == MetricType.MAHALANOBIS_DISTANCE:
            self._metric = MahalanobisDistance()
        else:
            raise NotImplementedError("Metric not implemented")
    
    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor) -> list[int]:
        return self._metric.rank(target_features, query_features)

