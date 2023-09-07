from cv2.typing import MatLike
import torch
from src.detectors import BaseDetector, DetectorResult
from src.embedders import BaseEmbedder
from src.helpers import draw_bbox

class TrackResults:
    def __init__(self, detect_result: DetectorResult, target_idx: int) -> None:
        self.detect_result = detect_result
        self.target_idx = target_idx

class BasePersonalTracker():
    def __init__(self, detector: BaseDetector, embedder: BaseEmbedder) -> None:
        self.detector = detector
        self.embedder = embedder
        self._target_features_pool: list[tuple[MatLike, torch.Tensor]] = []

    def track(self, frame: MatLike, draw_result: bool = False) -> TrackResults | None:
        detect_result = self.detector.detect(frame)
        if detect_result is None:
            return None
        detected_features = self.embedder.extract_features(frame, detect_result.bboxes)
        target_idx = self._max_average_cosine_distance(detected_features)
        if draw_result:
            self.draw_result(frame, TrackResults(detect_result, target_idx))
        return TrackResults(detect_result, target_idx)
    
    def add_target_features(self, cv_image: MatLike, bbox: tuple[int, int, int, int]) -> None:
        croped_image = cv_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        features = self.embedder.extract_feature(croped_image)
        self._target_features_pool.append((croped_image, features))
        
    def draw_result(self, frame: MatLike, track_result: TrackResults) -> None:
        detect_result = track_result.detect_result
        target_idx = track_result.target_idx

        for idx, bbox in enumerate(detect_result.bboxes):
            if idx == target_idx:
                draw_bbox(frame, bbox, (0, 255, 0), "Target") 
            else:
                draw_bbox(frame, bbox, (0, 0, 255))
            
    def _max_average_cosine_distance(self, detected_features: torch.Tensor) -> int:
        if len(detected_features) == 0:
            return -1
        max_average_cosine_distance = 0
        max_average_cosine_distance_index = -1
        avg_cosine_distance = torch.zeros(detected_features.shape[0])
        for i, (_, target_feature) in enumerate(self._target_features_pool):
            avg_cosine_distance += torch.nn.functional.cosine_similarity(target_feature.unsqueeze(0), detected_features)
        avg_cosine_distance /= len(self._target_features_pool) 
        max_average_cosine_distance = torch.max(avg_cosine_distance)
        max_average_cosine_distance_index = torch.argmax(avg_cosine_distance)
        return int(max_average_cosine_distance_index.item())
