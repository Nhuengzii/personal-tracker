from cv2.typing import MatLike
import torch
import cv2
import clip
from PIL import Image
from personal_tracker.embedder.emebedder import Embedder 


class CLIPEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    @torch.no_grad()
    def extract_feature(self, croped_image: MatLike) -> torch.Tensor:
        image = cv2.resize(croped_image, (224, 224))
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.preprocess(img).unsqueeze(0).to(self.device) # type: ignore
        image = image.to(self.device)
        feature: torch.Tensor = self.model.encode_image(image)
        return feature[0].cpu().type(torch.float32)
