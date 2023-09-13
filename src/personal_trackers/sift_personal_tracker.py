from typing import Any
from cv2.typing import MatLike
from src.detectors.base_detector import BaseDetector
from src.detectors.segment_detector import SegmentDetector
from src.embedders.base_embedder import BaseEmbedder
from src.metrics.base_metric import MetricType
from src.personal_trackers.personal_tracker import PersonalTracker
import cv2
import torch
from torch import Tensor

from src.personal_trackers.track_result import TrackResults


class SIFTPersonalTracker(PersonalTracker):
    def __init__(self, detector: SegmentDetector, embedder: BaseEmbedder, metric: MetricType, auto_add_target_features: bool = False, auto_add_target_features_interval: int = 60) -> None:
        super().__init__(detector, embedder, metric, auto_add_target_features, auto_add_target_features_interval)
        self.sift = cv2.SIFT_create() # type: ignore
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self._last_sift_features = None
        self._target_features_pool: list[tuple[MatLike, torch.Tensor, Any]] = []
        self._max_matches_threshold: int | None = None
        self._use_umbedder = False
        self._remove_query_background = False

    def _get_keypoints(self, img: MatLike):
        return self.sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)

    def _get_matches(self, target_keypoint, query_keypoints) -> list[int]:
        num_matches = []
        for q in query_keypoints:
            matches = self.bf.match(target_keypoint[1], q[1])
            matches = sorted(matches, key=lambda x: x.distance)
            num_matches.append(len(matches))
        return num_matches

    def _unsorted_scores(self, ranks: list[int], sorted_scores: list[float]) -> list[float]:
        unsorted_scores = [float(i) for i in range(len(ranks))]
        for idx, rank in enumerate(ranks):
            unsorted_scores[rank] = sorted_scores[idx]
        return unsorted_scores

    def use_embedder(self, use: bool = True):
        self._use_umbedder = use
        return self

    def remove_query_background(self, remove: bool = True):
        self._remove_query_background = remove
        return self

    def _algorithm(self, frame: MatLike) -> TrackResults:
        if self._use_umbedder:
            embedder_result = super()._algorithm(frame)
        else:
            detected_result = self.detector.detect(frame)
            embedder_result = TrackResults(detected_result)
        if embedder_result.detect_result is None:
            return TrackResults()
        detected_result = embedder_result.detect_result
        q_keypoints = []

        # Sift ranking
        for idx in range(len(detected_result.bboxes)):
            croped = detected_result.get_crop_and_remove_background(idx, self._remove_query_background)
            q_keypoints.append(self._get_keypoints(croped))
        matches = [0 for _ in range(len(q_keypoints))]
        for idx in range(len(self._target_features_pool)):
            ms = self._get_matches(self._target_features_pool[idx][2], q_keypoints)
            assert len(ms) == len(matches)
            for i in range(len(ms)):
                matches[i] += ms[i]
        if self._max_matches_threshold is None:
            self._max_matches_threshold = int(max(matches) * 0.8)
        # if max(matches) < self._max_matches_threshold:
        #     print(f"max matches: {max(matches)} is below threshold: {self._max_matches_threshold}")
        #     return None
        if self._last_sift_features is not None:
            ms = self._get_matches(self._last_sift_features, q_keypoints)
            assert len(ms) == len(matches)
            for i in range(len(ms)):
                matches[i] += ms[i]
        print(matches)

        assert all([match > 0 for match in matches]), "match should be greater than 0"
        assert self.metric.metric_type == MetricType.COSINE_SIMILARITY, "Available only for cosine similarity"
        if not embedder_result.success:
            sorted_scores, ranks = torch.sort(torch.tensor(matches), descending=True) # from high to low
        else:
            assert embedder_result.ranks is not None
            assert embedder_result.sorted_scores is not None
            embedder_scores = self._unsorted_scores(embedder_result.ranks, embedder_result.sorted_scores)
            combined_scores = [2 * (matches[i] * embedder_scores[i]) / (matches[i] + embedder_scores[i]) for i in range(len(matches))]
            sorted_scores, ranks = torch.sort(torch.tensor(combined_scores), descending=True) # from high to low

        result = TrackResults(detected_result, ranks[0], ranks.tolist(), sorted_scores.tolist())
        assert result.target_idx is not None
        croped = detected_result.get_crop_and_remove_background(result.target_idx)
        self._last_sift_features = self._get_keypoints(croped)
        return result


    def draw_result(self, frame: MatLike, track_result: TrackResults) -> None:
        super().draw_result(frame, track_result)
        # assert track_result.target_idx is not None
        # assert track_result.detect_result is not None
        # target_crop = track_result.detect_result.get_crop_and_remove_background(track_result.target_idx)
        # cv2.imshow("target_crop", target_crop)

    def add_target_features(self, cv_image: MatLike, bbox: tuple[int, int, int, int]) -> None:
        croped_image = cv_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        detect_result = self.detector.detect(croped_image)
        assert detect_result is not None, "detect_result should not be None"
        # assert detect_result.num_detected == 1, "detect_result.num_detected should be 1"
        rm_bg = detect_result.get_crop_and_remove_background(0)
        kp, des = self._get_keypoints(rm_bg)
        features = self.embedder.extract_feature(croped_image)
        self._target_features_pool.append((croped_image, features, (kp, des)))
