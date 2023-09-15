from torch import Tensor
from personal_tracker.detector.detector_result import DetectorResult

class TrackResult:
    def __init__(self, detect_result: DetectorResult | None = None, target_idx: int | None = None, ranks: list[int] | None = None, sorted_scores: list[float] | None = None, target_features: Tensor | None = None, detected_results: Tensor | None = None ) -> None:
        self.detect_result = detect_result
        self.target_idx = target_idx
        self.ranks = ranks
        self.sorted_scores = sorted_scores
        self.target_features = target_features
        self.detected_features = detected_results
        self._k_target_bbox: tuple[int, int, int, int] | None = None

    @property
    def success(self) -> bool:
        return self.target_bbox is not None
            

    @property
    def target_bbox(self) -> tuple[int, int, int, int] | None:
        if self.is_overwrited:
            assert self._k_target_bbox is not None
            return self._k_target_bbox
        if self.target_idx is None:
            return None
        assert self.detect_result is not None
        return self.detect_result.bboxes[self.target_idx]

    @property
    def target_score(self) -> float:
        assert self.target_idx is not None
        assert self.detect_result is not None
        assert self.sorted_scores is not None
        return self.sorted_scores[self.target_idx]

    @property
    def is_overwrited(self) -> bool:
        return self._k_target_bbox is not None

    def k_overwrite(self, bbox: tuple[int, int, int, int] ) -> None:
        self._k_target_bbox = bbox
