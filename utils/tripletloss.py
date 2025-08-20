import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Combined loss function for regression tasks that balances:
        - RMSE (Root Mean Squared Error) -> absolute accuracy
        - Optional centered cosine similarity (proxy for Pearson in small batches)
        - Pearson correlation loss -> relative ranking preservation

    Designed for multi-target regression (B × D output), such as predicting
    binding scores for multiple proteins at once.

    Key design choices:
    -------------------
    1. Multi-target safe:
       Computes Pearson (and cosine) **per target column** and averages.
       This ensures each target's correlation contributes equally to the loss.

    2. No cosine/Pearson duplication:
       In the original version, cosine was computed after mean-centering, which
       makes it mathematically identical to Pearson. Here, you can choose:
         - Use Pearson only (recommended for large batches)
         - Or use cosine without centering (different from Pearson) as an optional proxy.

    3. Numerical stability:
       Variance and norm terms are clamped with eps to prevent division by zero.
       Handles cases where predictions collapse to a constant vector.

    Parameters
    ----------
    alpha : float
        Weight for RMSE term (absolute error).
    beta : float
        Weight for cosine similarity loss (set 0 to disable).
    gamma : float
        Weight for Pearson correlation loss.
    eps : float
        Small constant to avoid division by zero in normalization.

    Example
    -------
    >>> loss_fn = TripletLoss(alpha=0.1, beta=0.0, gamma=1.5)
    >>> loss = loss_fn(preds, targets)
    """

    def __init__(self, alpha=0.1, beta=0.0, gamma=1.5, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds, targets):
        """
        Compute the combined loss.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions, shape (B,) or (B, D).
        targets : torch.Tensor
            Ground truth values, same shape as preds.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Ensure preds and targets are the same shape
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")

        # --- RMSE term ---
        rmse = torch.sqrt(F.mse_loss(preds, targets) + self.eps)

        # --- Cosine similarity loss (optional) ---
        if self.beta != 0:
            # Normalize along batch dimension for each target separately
            cos_loss = 1.0 - F.cosine_similarity(
                preds, targets, dim=0 if preds.dim() == 1 else 0
            ).mean()  # mean over targets if multi-output
        else:
            cos_loss = torch.tensor(0.0, device=preds.device)

        # --- Pearson correlation loss ---
        if preds.dim() == 1:
            # Single target: center, compute correlation
            x = preds - preds.mean()
            y = targets - targets.mean()
            vx = torch.clamp((x ** 2).mean(), min=self.eps)
            vy = torch.clamp((y ** 2).mean(), min=self.eps)
            pearson = (x * y).mean() / torch.sqrt(vx * vy)
            pearson_loss = 1.0 - pearson.clamp(-1.0, 1.0)
        else:
            # Multi-target: compute per column
            x = preds - preds.mean(dim=0, keepdim=True)
            y = targets - targets.mean(dim=0, keepdim=True)
            vx = torch.clamp((x ** 2).mean(dim=0), min=self.eps)
            vy = torch.clamp((y ** 2).mean(dim=0), min=self.eps)
            corr = (x * y).mean(dim=0) / torch.sqrt(vx * vy)
            pearson_loss = (1.0 - corr.clamp(-1.0, 1.0)).mean()

        # --- Final weighted sum ---
        total = self.alpha * rmse + self.beta * cos_loss + self.gamma * pearson_loss
        return total
