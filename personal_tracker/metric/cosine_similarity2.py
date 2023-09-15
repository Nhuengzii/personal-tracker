import torch
class CosineSimilarity2:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def compute_similarity(self, target_features: torch.Tensor, query_features: torch.Tensor):
        means = target_features.mean(dim=0)
        sims = torch.cosine_similarity(query_features, means.unsqueeze(0), dim=1)
        return sims

    @torch.no_grad()
    def rank(self, target_features: torch.Tensor, query_features: torch.Tensor) -> tuple[list[int], list[float]]:
        similarity = self.compute_similarity(target_features, query_features)
        sorted_similarity, indices = torch.sort(similarity, descending=True)
        return indices.tolist(), sorted_similarity.tolist()
