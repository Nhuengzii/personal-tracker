from src.detectors.detector_result import DetectorResult
from torch import Tensor


class TrackResults:
    def __init__(self, detect_result: DetectorResult, target_idx: int, ranks: list[int], sorted_scores: list[float], target_features: Tensor| None = None, detected_results: Tensor | None = None ) -> None:
        self.detect_result = detect_result
        self.target_idx = target_idx
        self.ranks = ranks
        self.sorted_scores = sorted_scores
        self.target_features = target_features
        self.detected_features = detected_results
        self.overwrited = False
        self.overwrited_bbox: tuple[int, int, int, int] | None = None

    @property
    def target_bbox(self) -> tuple[int, int, int, int]:
        if self.overwrited:
            assert self.overwrited_bbox is not None
            return self.overwrited_bbox
        return self.detect_result.bboxes[self.target_idx]

    def overwrite(self, bbox: tuple[int, int, int, int] ) -> None:
        self.is_desprecated = True
        self.overwrited_bbox = bbox
