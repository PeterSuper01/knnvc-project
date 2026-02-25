import torch
from torch import Tensor


def fast_cosine_dist(
    source_feats: Tensor, matching_pool: Tensor, device: str = "cpu"
) -> Tensor:
    """
    Like torch.cdist, but fixed dim=-1 and for cosine distance.
    source_feats: [B, D]
    matching_pool: [M, D]
    returns: [B, M]
    """
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = (
        -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0] ** 2
        + source_norms[:, None] ** 2
        + matching_norms[None] ** 2
    )
    dotprod /= 2

    dists = 1 - (dotprod / (source_norms[:, None] * matching_norms[None]))
    return dists


def knn_search(
    source_feats: Tensor, matching_pool: Tensor, k: int, device: str = "cpu"
) -> Tensor:
    dists = fast_cosine_dist(source_feats, matching_pool, device)
    _, indices = dists.topk(k, dim=-1, largest=False)
    # pred_source_feats = torch.zeros_like(source_feats)
    # for i in range(source_feats.shape[0]):
    #     pred_source_feats[i, :] = matching_pool[indices[i, :], :].mean(dim=0)
    pred_source_feats = matching_pool[indices].mean(dim=1)
    return pred_source_feats
