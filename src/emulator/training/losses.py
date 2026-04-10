import torch
import torch.nn as nn


class LatitudeWeightedL1(nn.Module):
    """Latitude-weighted L1 loss used for deterministic stage-1 training."""

    def __init__(self, h=64):
        super().__init__()
        lats = 90.0 - (torch.arange(h) + 0.5) * (180.0 / h)
        lats_rad = torch.deg2rad(lats)
        weights = h * torch.cos(lats_rad) / torch.sum(torch.cos(lats_rad))
        self.register_buffer("weights", weights.view(1, 1, h, 1))

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target) * self.weights)


class Stage2Loss(nn.Module):
    """CRPS + KL loss used for probabilistic joint training."""

    def __init__(self, ensemble_size=4, kl_weight=1e-4):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.kl_weight = kl_weight

    def forward(self, preds_flat, mu_p, logvar_p, mu_q, logvar_q, target):
        b, c, h, w = target.shape
        preds = preds_flat.view(self.ensemble_size, b, c, h, w)
        target_expanded = target.unsqueeze(0)

        abs_err = torch.abs(preds - target_expanded)
        term1 = torch.mean(abs_err, dim=0)

        pairwise_diff = torch.abs(preds.unsqueeze(1) - preds.unsqueeze(0))
        term2 = 0.5 * torch.mean(pairwise_diff, dim=(0, 1))

        crps = torch.mean(term1 - term2)

        kl_div = 0.5 * torch.sum(
            logvar_p
            - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
            - 1,
            dim=[1, 2, 3],
        )
        kl_div = torch.mean(kl_div)

        total_loss = crps + (self.kl_weight * kl_div)
        return total_loss, crps, kl_div
