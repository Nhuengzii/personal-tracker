from cv2.typing import MatLike
import torch
from src.embedders.available_embedder_models import AvailableEmbedderModels
from src.models import osnet_ain, osnet
import numpy as np
import cv2


class BaseEmbedder:
    def __init__(self, model: AvailableEmbedderModels = AvailableEmbedderModels.OSNET_AIN_X1_0) -> None:
        if model == AvailableEmbedderModels.OSNET_AIN_X1_0:
            self.model = osnet_ain.osnet_ain_x1_0(pretrained=True)
        elif model == AvailableEmbedderModels.OSNET_X1_0:
            self.model = osnet.osnet_x1_0(pretrained=True)
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
