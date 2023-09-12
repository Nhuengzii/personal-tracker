import torch
from src.metrics.cosine_similarity import CosineSimilarity
from src.metrics.euclidian import EuclideanDistance
from src.metrics.mahalanobis import MahalanobisDistance


class CSEMMetric():
    def __init__(self) -> None:
        self.cosine_similarity = CosineSimilarity()
        self.euclidean_distance = EuclideanDistance()
        self.mahalanobis_distance = MahalanobisDistance()

    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor, kalman_distances: torch.Tensor | None = None) -> tuple[list[int], list[float]]:
        cs_scores = self.cosine_similarity.compute_similarity(target_features, query_features) + 1
        inv_cs_scores = 1 / cs_scores
        inv_cs_scores_means = inv_cs_scores.T.mean(dim=1)
        normalized_inv_cs_scores = (inv_cs_scores_means - inv_cs_scores_means.mean()) / inv_cs_scores_means.std()
        e_scores = self.euclidean_distance.compute_distance(target_features, query_features)
        e_scores_means = torch.mean(e_scores.T, dim=1)
        normalized_e_scores = (e_scores_means - e_scores_means.mean()) / e_scores_means.std()
        m_scores = self.mahalanobis_distance.compute_distances(target_features, query_features)
        m_scores_means = torch.mean(m_scores, dim=1)
        normalized_m_scores = (m_scores_means - m_scores_means.mean()) / m_scores_means.std()
        scores = normalized_inv_cs_scores + normalized_e_scores + normalized_m_scores
        scores = normalized_inv_cs_scores + normalized_m_scores
        sorted_scores, sorted_indices = torch.sort(scores)
        return sorted_indices.cpu().numpy().tolist(), sorted_scores.cpu().numpy().tolist()
