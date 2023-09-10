from src.metrics.base_metric import BaseMetric, MetricType
import torch
from torch import nn, optim


class BiasingMetric(BaseMetric):
    def __init__(self, metric: MetricType = MetricType.COSINE_SIMILARITY) -> None:
        super().__init__(metric)
        self.biasor_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.device = "cpu"
        self.optimizer = optim.Adam(self.biasor_model.parameters(), lr=0.001)

    def _init_biasor_model(self, init_features: torch.Tensor):
        for i in range(10):
            pred = self.biasor_model(init_features)
            loss = torch.corrcoef

