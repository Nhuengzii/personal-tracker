import torch
from .cosine_similarity import CosineSimilarity
from .euclidian import EuclideanDistance
from .mahalanobis import MahalanobisDistance

class CSEM():
    def __init__(self) -> None:
        self.cosine_similarity = CosineSimilarity()
        self.euclidean_distance = EuclideanDistance()
        self.mahalanobis_distance = MahalanobisDistance()

    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor) -> tuple[list[int], list[float]]:
        assert len(query_features) > 0
        cs_scores = self.cosine_similarity.compute_similarity(target_features, query_features) + 1
        e_scores = self.euclidean_distance.compute_distance(target_features, query_features)
        m_scores = self.mahalanobis_distance.compute_distances(target_features, query_features)
        if len(query_features) == 1:
            mean_cs_scores = cs_scores.T.mean(dim=1)
            mean_e_scores = e_scores.T.mean(dim=1)
            mean_m_scores = m_scores.mean(dim=1)
            scores = mean_cs_scores + mean_e_scores + mean_m_scores
            sorted_scores, sorted_indices = torch.sort(scores)
            return sorted_indices.cpu().numpy().tolist(), sorted_scores.cpu().numpy().tolist()
            
        inv_cs_scores = 1 / cs_scores
        inv_cs_scores_means = inv_cs_scores.T.mean(dim=1)
        normalized_inv_cs_scores = (inv_cs_scores_means - inv_cs_scores_means.mean()) / inv_cs_scores_means.std()
        e_scores_means = torch.mean(e_scores.T, dim=1)
        normalized_e_scores = (e_scores_means - e_scores_means.mean()) / e_scores_means.std()
        m_scores_means = torch.mean(m_scores, dim=1)
        normalized_m_scores = (m_scores_means - m_scores_means.mean()) / m_scores_means.std()
        scores = normalized_inv_cs_scores + normalized_e_scores + normalized_m_scores
        scores = normalized_inv_cs_scores + normalized_m_scores
        sorted_scores, sorted_indices = torch.sort(scores)
        return sorted_indices.cpu().numpy().tolist(), sorted_scores.cpu().numpy().tolist()
