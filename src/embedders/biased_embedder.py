from cv2.typing import MatLike
from src.embedders.available_embedder_models import AvailableEmbedderModels
from src.embedders.base_embedder import BaseEmbedder
from torch import nn
import torch
import os

class BiasedEmbedder(BaseEmbedder):
    def __init__(self, model: AvailableEmbedderModels) -> None:
        super().__init__(model)
        self.biasor_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        cwd = os.getcwd()
        self.biasor_model.load_state_dict(torch.load(f"{cwd}/biasor.pth"))
    
    @torch.no_grad()
    def extract_feature(self, croped_image: MatLike) -> torch.Tensor:
        features =  super().extract_feature(croped_image)
        return self.biasor_model(features.unsqueeze(0))[0]
