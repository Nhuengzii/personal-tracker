import torch

class MahalanobisDistance():
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def compute_distances(self, target_features: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mahalanobis distance between the target and query features.
        Mahalanobis distance is a measure of the distance between a point P and a distribution D.
        The formula of the Mahalanobis distance is:
            D_M = sqrt((P - M)^T * C^-1 * (P - M))
        where:
            P is the query feature
            M is the mean vector of the target features
            C is the covariance matrix of the target features
        Args:
            target_features (torch.Tensor): target features with shape (num_target_samples, feature_dim)
            query_features (torch.Tensor): query features with shape (num_query_samples, feature_dim)
        Returns:
            torch.Tensor: Mahalanobis distance between the target and query features with shape (num_query_samples, )
        """

        cov = torch.cov(target_features.T) + torch.eye(target_features.shape[1]) * 1e-5
        cov_inv = torch.inverse(cov)

        diff = target_features - query_features.unsqueeze(1)
        l = torch.matmul(diff, cov_inv)
        dist = torch.sqrt(torch.sum(l * diff, dim=-1))

        return dist
    
    @torch.no_grad()
    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor) -> tuple[list[int], list[float]]:
        """
        Rank the query features based on the Mahalanobis distance to the target features.
        Args:
            target_features (torch.Tensor): target features with shape (num_target_samples, feature_dim)
            query_features (torch.Tensor): query features with shape (num_query_samples, feature_dim)
        Returns:
            list[int]: list of indices that sort the query features based on the Mahalanobis distance to the target features
            ordered from most similar to least similar
        """

        dist = self.compute_distances(target_features, query_features)
        mean_dist = torch.mean(dist, dim=1)
        sorted_dist, indices = torch.sort(mean_dist)
        return indices.tolist(), sorted_dist.tolist()
