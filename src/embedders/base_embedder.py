from cv2.typing import MatLike
import torch
from src.models.osnet_ain import osnet_ain_x1_0
import numpy as np
import cv2


class BaseEmbedder:
    def __init__(self) -> None:
        self.model = osnet_ain_x1_0(pretrained=True)
        self.model.eval()
        self.image_size = (128, 256)
    
    @torch.no_grad()
    def extract_feature(self, croped_image: MatLike) -> torch.Tensor:
        image = cv2.resize(croped_image, self.image_size)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)
        feature = self.model(image)
        return feature[0]

    @torch.no_grad()
    def extract_features(self, frame: MatLike, bboxes: list[tuple[int, int, int, int]]) -> torch.Tensor:
        features = []
        for bbox in bboxes:
            croped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            feature = self.extract_feature(croped_image)
            features.append(feature)
        return torch.stack(features)
