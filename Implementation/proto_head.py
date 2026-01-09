import torch
import torch.nn as nn

def classwise_mean_cov(supp_emb: torch.Tensor, supp_y: torch.Tensor, n_way: int, shrink: float = 0.1):
    """
    Return class means [n_way,D] and Cholesky factors L of covariances [n_way,D,D]
    such that cov = L @ L^T (after shrinkage).
    """
    device = supp_emb.device
    n, d = supp_emb.shape
    means = torch.zeros(n_way, d, device=device)
    covs  = torch.zeros(n_way, d, d, device=device)
    eps = 1e-5

    eye = torch.eye(d, device=device)
    for c in range(n_way):
        x = supp_emb[supp_y == c]
        mu = x.mean(dim=0, keepdim=True)
        xm = x - mu
        cov = (xm.t() @ xm) / max(1, x.shape[0] - 1) + eps * eye
        cov = (1.0 - shrink) * cov + shrink * eye
        means[c] = mu
        covs[c]  = cov

    L = torch.linalg.cholesky(covs)  # [n_way, d, d]
    return means, L

class AdaptiveProtoHead(nn.Module):
    def __init__(self, temperature: float = 1.0, shrinkage: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.shrinkage = shrinkage

    @staticmethod
    def weighted_mean(emb: torch.Tensor, labels: torch.Tensor, n_way: int, w: torch.Tensor):
        """Return weighted means per class. emb:[N,D], labels:[N], w:[N]."""
        device = emb.device
        n, d = emb.shape
        means = torch.zeros(n_way, d, device=device)
        for c in range(n_way):
            mask = (labels == c)
            wc = w[mask].clamp(min=1e-3).unsqueeze(-1)  # [k,1]
            xc = emb[mask]                               # [k,D]
            means[c] = (wc * xc).sum(dim=0) / wc.sum(dim=0)
        return means

    def forward(self, supp_emb, supp_y, qry_emb, n_way, reliability):
        means_w = self.weighted_mean(supp_emb, supp_y, n_way, reliability)  # [n_way,D]
        means, L = classwise_mean_cov(supp_emb, supp_y, n_way, self.shrinkage)

        # Mahalanobis distance via triangular solve: solve L y = (x - μ) ; dist = ||y||^2
        diff = qry_emb.unsqueeze(1) - means_w.unsqueeze(0)  # [Nq, n_way, D]
        # reshape for batched solve: (n_way batches)
        Nq, C, D = diff.shape
        diff2 = diff.permute(1,0,2).reshape(C, Nq, D)       # [n_way, Nq, D]
        y = torch.linalg.solve_triangular(L, diff2.transpose(1,2), upper=False)   # [n_way, D, Nq]
        # now solve L^T z = y  ⇒ but for norm^2 we can do sum of squares of y after both solves:
        z = torch.linalg.solve_triangular(L.transpose(-1,-2), y, upper=True)      # [n_way, D, Nq]
        d2 = (z**2).sum(dim=1).transpose(0,1)  # [Nq, n_way]
        logits = -d2 / self.temperature
        return logits, means_w
