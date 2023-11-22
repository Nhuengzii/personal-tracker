from enum import Enum

class MetricType(Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    MAHALANOBIS_DISTANCE = "mahalanobis_distance"
    CSEM_DISTANCE = "csem_distance"
