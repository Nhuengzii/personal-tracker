import torch
class CosineSimilarity:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def compute_similarity(self, target_features: torch.Tensor, query_features: torch.Tensor):
        """
        Compute the cosine similarity between target and query features

        Args:
            target_features (torch.Tensor): target features with shape (num_target_features, feature_dim)
            query_features (torch.Tensor): query features with shape (num_query_features, feature_dim)
        
        Returns:
            torch.Tensor: cosine similarity between target and query features with shape (num_target_features, num_query_features)
            each element (i, j) represents the cosine similarity between target[i] and query[j]
        """

        ret = []
        for i in target_features:
            ret.append(torch.cosine_similarity(i, query_features))
        return torch.stack(ret)

    @torch.no_grad()
    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor) -> tuple[list[int], list[float]]:
        """
        Rank the query features based on the cosine similarity to the target features.

        Args:
            target_features (torch.Tensor): target features with shape (num_target_samples, feature_dim)
            query_features (torch.Tensor): query features with shape (num_query_samples, feature_dim)
        
        Returns:
            list[int]: list of indices that sort the query features based on the cosine similarity to the target features
            ordered from most similar to least similar
        """

        similarity = self.compute_similarity(target_features, query_features)
        mean_similarity = torch.mean(similarity.T, dim=1)
        sorted_similarity, indices = torch.sort(mean_similarity, descending=True)
        return indices.tolist(), sorted_similarity.tolist()
