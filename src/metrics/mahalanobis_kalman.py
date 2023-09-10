from ultralytics.engine.results import torch
from src.metrics.mahalanobis import MahalanobisDistance


class MahalanobisKalmanDistance(MahalanobisDistance):
    @torch.no_grad()
    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor, kalman_distances: torch.Tensor | None = None) -> tuple[list[int], list[float]]:
        dist = self.compute_distances(target_features, query_features)
        mean_dist = torch.mean(dist, dim=1)
        if kalman_distances is None:
            return torch.argsort(mean_dist).tolist(), mean_dist.tolist()
        norm_dist = mean_dist / mean_dist.sum()
        kalman_distances = kalman_distances / kalman_distances.sum()
        kalman_weight = 0.2
        combined_dist = kalman_weight * kalman_distances + (1 - kalman_weight) * norm_dist
        sorted_dist, indices = torch.sort(combined_dist)
        return indices.tolist(), sorted_dist

