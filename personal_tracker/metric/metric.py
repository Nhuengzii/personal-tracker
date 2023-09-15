import torch

from personal_tracker.metric.cosine_similarity2 import CosineSimilarity2
from .cosine_similarity import CosineSimilarity
from .csem import CSEM
from .euclidian import EuclideanDistance
from .mahalanobis import MahalanobisDistance
from .metric_type import MetricType

class Metric:
    def __init__(self, metric_type: MetricType = MetricType.COSINE_SIMILARITY) -> None:
        self.metric_type = metric_type
        if metric_type == MetricType.COSINE_SIMILARITY:
            self._metric = CosineSimilarity()
        elif metric_type == MetricType.MAHALANOBIS_DISTANCE:
            self._metric = MahalanobisDistance()
        elif metric_type == MetricType.EUCLIDEAN_DISTANCE:
            self._metric = EuclideanDistance()
        elif metric_type == MetricType.CSEM_DISTANCE:
            self._metric = CSEM()
        elif metric_type == MetricType.COSINE_SIMILARITY2:
            self._metric = CosineSimilarity2()
        else:
            raise NotImplementedError(f"Metric type {metric_type} is not implemented")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor) -> tuple[list[int], list[float]]:
        return self._metric.rank(target_features, query_features)
