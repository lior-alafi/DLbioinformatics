import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn.functional import cosine_similarity

class TripletLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rmse = MSELoss()

    def forward(self, preds, targets):
        # RMSE
        mse = self.rmse(preds, targets)
        rmse = torch.sqrt(mse + 1e-8)

        # Cosine similarity (want high value, so we add it directly)
        cos = cosine_similarity(preds.unsqueeze(0), targets.unsqueeze(0)).mean()

        # Pearson correlation (centered dot product)
        pred_centered = preds - preds.mean()
        targ_centered = targets - targets.mean()

        numerator = (pred_centered * targ_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (targ_centered ** 2).sum() + 1e-8)
        pearson = numerator / denominator

        # Final loss
        loss = (self.alpha * rmse + self.beta * cos + self.gamma * (1 - pearson))**2
        return loss
