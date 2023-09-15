from personal_tracker.detector.detector import Detector
from personal_tracker.embedder.available_embedder_models import AvailableEmbedderModels
from personal_tracker.metric.metric_type import MetricType


class TrackerConfig:
    def __init__(self) -> None:
        self.embedder_model: AvailableEmbedderModels | None = None
        self.metric_type: MetricType = MetricType.COSINE_SIMILARITY
        self.sift_history_size: int = 1
        self.auto_add_target_features: bool = False
        self.auto_add_target_features_interval: int = 99999999999

    def set_metric_type(self, metric_type: MetricType):
        self.metric_type = metric_type
        return self

    def set_embedder_model(self, model: AvailableEmbedderModels):
        self.embedder_model = model
        return self

    def set_sift_history_size(self, size: int):
        self.sift_history_size = size
        return self

    def set_auto_add_target_features(self, auto_add: bool = False, interval: int = 99999999999):
        if auto_add and interval == 99999999999:
            print("Warning: please set interval for auto add target features")
        self.auto_add_target_features = auto_add
        self.auto_add_target_features_interval = interval
        return self
