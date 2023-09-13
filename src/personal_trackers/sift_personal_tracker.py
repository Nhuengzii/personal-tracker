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

    def _algorithm(self, frame: MatLike) -> TrackResults | None:
        raw_result =  super()._algorithm(frame)
        if raw_result is None:
            return None
        # if len(raw_result.detect_result.bboxes) == 1:
        #     return raw_result
        if self._last_sift_features is None:
            croped = raw_result.detect_result.get_crop_and_remove_background(raw_result.target_idx )
            self._last_sift_features = self._get_keypoints(croped)
            return raw_result
        q_keypoints = []

        # Sift ranking
        for idx in range(len(raw_result.detect_result.bboxes)):
            croped = raw_result.detect_result.get_crop_and_remove_background(idx)
            q_keypoints.append(self._get_keypoints(croped))
        matches = self._get_matches(self._last_sift_features, q_keypoints)
        for idx in range(len(self._target_features_pool)):
            ms = self._get_matches(self._target_features_pool[idx][2], q_keypoints)
            assert len(ms) == len(matches)
            for i in range(len(ms)):
                matches[i] += ms[i]
            
        assert all([match > 0 for match in matches]), "match should be greater than 0"
        # inv_matches = [1 / match for match in matches]
        # mean_matches = sum(inv_matches) / len(inv_matches)
        # std_matches = sum([(match - mean_matches) ** 2 for match in inv_matches]) / len(inv_matches)
        # assert std_matches != 0, "std_matches should not be 0"
        # normalized_matches = [(match - mean_matches) / std_matches for match in inv_matches]
        sorted_matches, ranks = torch.sort(torch.tensor(matches), descending=True) # from large to small
        # if self._max_matches_threshold is None and len(sorted_matches) == 1:
        #     self._max_matches_threshold = int(sorted_matches[ranks[0]] * 0.75)
        # else:
        #     assert self._max_matches_threshold is not None
        #     if float(sorted_matches[ranks[0]]) < self._max_matches_threshold:
        #         print(f"max_matches_threshold is too low. below {self._max_matches_threshold}")
        #         return None
        #         return raw_result
        print(sorted_matches)
        result = TrackResults(raw_result.detect_result, ranks[0], ranks, sorted_matches, raw_result.target_features, raw_result.detected_features)
        return result
        
        raw_ranks = raw_result.ranks
        raw_sorted_scores = raw_result.sorted_scores
        raw_unsorted_scores = self._unsorted_scores(raw_ranks, raw_sorted_scores)
        raw_mean_score = sum(raw_unsorted_scores) / len(raw_unsorted_scores)
        raw_std_score = sum([(score - raw_mean_score) ** 2 for score in raw_unsorted_scores]) / len(raw_unsorted_scores)
        assert raw_std_score != 0, "std_score should not be 0"
        raw_normalized_score = [(score - raw_mean_score) / raw_std_score for score in raw_unsorted_scores]
        
        sum_scores = [normalized_matches[idx] + raw_normalized_score[idx] for idx in range(len(normalized_matches))]
        sorted_sum_scores, ranks = torch.sort(torch.tensor(sum_scores), descending=False) # from small to large
        result = TrackResults(raw_result.detect_result, ranks[0], ranks, sorted_sum_scores)
        return result

    def draw_result(self, frame: MatLike, track_result: TrackResults) -> None:
        super().draw_result(frame, track_result)
        target_crop = track_result.detect_result.get_crop_and_remove_background(track_result.target_idx)
        cv2.imshow("target_crop", target_crop)

    def add_target_features(self, cv_image: MatLike, bbox: tuple[int, int, int, int]) -> None:
        croped_image = cv_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        detect_result = self.detector.detect(croped_image)
        assert detect_result is not None, "detect_result should not be None"
        # assert detect_result.num_detected == 1, "detect_result.num_detected should be 1"
        rm_bg = detect_result.get_crop_and_remove_background(0)
        kp, des = self._get_keypoints(rm_bg)
        features = self.embedder.extract_feature(croped_image)
        self._target_features_pool.append((croped_image, features, (kp, des)))
