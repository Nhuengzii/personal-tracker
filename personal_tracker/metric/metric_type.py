from enum import Enum

class MetricType(Enum):
    COSINE_SIMILARITY = 0
    EUCLIDEAN_DISTANCE = 1
    MAHALANOBIS_DISTANCE = 2
    CSEM_DISTANCE = 3
