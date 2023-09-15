from cv2.typing import MatLike
import torch
import torchvision
import numpy as np
import cv2

from personal_tracker.embedder.available_embedder_models import AvailableEmbedderModels
from personal_tracker.models import osnet, osnet_ain


class Embedder:
    def __init__(self, model: AvailableEmbedderModels = AvailableEmbedderModels.OSNET_AIN_X1_0) -> None:
        if model == AvailableEmbedderModels.OSNET_AIN_X1_0:
            self.model = osnet_ain.osnet_ain_x1_0(pretrained=True)
        elif model == AvailableEmbedderModels.OSNET_X1_0:
            self.model = osnet.osnet_x1_0(pretrained=True)
        elif model == AvailableEmbedderModels.OSNET_AIN_X0_75:
            self.model = osnet_ain.osnet_ain_x0_75(pretrained=True)
        elif model == AvailableEmbedderModels.OSNET_X0_75:
            self.model = osnet.osnet_x0_75(pretrained=True)
        elif model == AvailableEmbedderModels.OSNET_AIN_X0_5:
            self.model = osnet_ain.osnet_ain_x0_5(pretrained=True)
        elif model == AvailableEmbedderModels.OSNET_X0_5:
            self.model = osnet.osnet_x0_5(pretrained=True)
        elif model == AvailableEmbedderModels.OSNET_AIN_X0_25:
            self.model = osnet_ain.osnet_ain_x0_25(pretrained=True)
        elif model == AvailableEmbedderModels.OSNET_X0_25:
            self.model = osnet.osnet_x0_25(pretrained=True)
        self.model.eval()
        self.image_size = (128, 256)
        self.preprocessor = torchvision.transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    @torch.no_grad()
    def extract_feature(self, croped_image: MatLike) -> torch.Tensor:
        image = cv2.resize(croped_image, self.image_size)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        image = self.preprocessor(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        feature: torch.Tensor = self.model(image)
        return feature[0].cpu()

    @torch.no_grad()
    def extract_features(self, frame: MatLike, bboxes: list[tuple[int, int, int, int]]) -> torch.Tensor:
        features = []
        for bbox in bboxes:
            croped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            feature = self.extract_feature(croped_image)
            features.append(feature)
        return torch.stack(features)
