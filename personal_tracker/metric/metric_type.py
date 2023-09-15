from enum import Enum

class MetricType(Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    COSINE_SIMILARITY2 = "cosine_similarity2"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    MAHALANOBIS_DISTANCE = "mahalanobis_distance"
    CSEM_DISTANCE = "csem_distance"
