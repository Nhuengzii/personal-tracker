import torch
class EuclideanDistance():
    def __init__(self) -> None:
        pass

    def compute_distance(self, target_features: torch.Tensor, query_features: torch.Tensor):
        """
        Compute the euclidean distance between target and query features

        Args:
            target_features (torch.Tensor): target features with shape (batch_size, feature_dim)
            query_features (torch.Tensor): query features with shape (batch_size, feature_dim)

        Returns:
            torch.Tensor: euclidean distance between target and query features with shape (batch_size, batch_size)
            each element (i, j) represents the euclidean distance between target[i] and query[j]
        """
        return torch.cdist(target_features, query_features, p=2)

    @torch.no_grad()
    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor) -> tuple[list[int], list[float]]:
        dist = self.compute_distance(target_features, query_features)
        mean_dist = torch.mean(dist.T, dim=1)
        sorted_dist, indices = torch.sort(mean_dist)
        return indices.tolist(), sorted_dist.tolist()
