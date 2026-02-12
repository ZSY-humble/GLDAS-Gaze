import torch
import torch.nn as nn
import torch.nn.functional as F
class FixationKLLoss(nn.Module):
    """KL divergence loss for fixation prediction."""

    def __init__(self, temperature=1.0, epsilon=1e-8, scale=1):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.scale = scale

    def forward(self, pred, target, alpha=None, beta=None, weights=None):
        bs = pred.size(0)
        pred_flat = pred.view(bs, -1)
        target_flat = target.view(bs, -1)
        pred_log_prob = F.log_softmax(pred_flat / self.temperature, dim=1)
        target_prob = target_flat / (target_flat.sum(dim=1, keepdim=True) + self.epsilon)
        kl_loss = F.kl_div(pred_log_prob, target_prob, reduction='none').sum(dim=1)
        if weights is not None:
            weight_factor = weights.view(bs, -1).mean(dim=1)
            kl_loss = kl_loss * weight_factor
            
        return kl_loss.mean()*self.scale

class CombinedFixationLoss(nn.Module):
    """BCE + KL fixation loss."""
    def __init__(self, bce_weight=0.7, kl_weight=0.3, use_focal=False):
        super().__init__()
        self.bce_weight = bce_weight
        self.kl_weight = kl_weight
        self.use_focal = use_focal
        self.kl_loss = FixationKLLoss()
        
        if use_focal:
            from common.losses import focal_loss
            self.bce_loss = focal_loss
        else:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred, target, alpha=1, beta=4, weights=None):
        bs = pred.size(0)
        if self.use_focal:
            bce_loss = self.bce_loss(pred, target, alpha, beta, weights)
        else:
            bce_loss = self.bce_loss(pred, target)
            if weights is not None:
                bce_loss = (bce_loss * weights).view(bs, -1).mean(dim=1)
            bce_loss = bce_loss.mean()
        kl_loss = self.kl_loss(pred, target, weights)
        total_loss = self.bce_weight * bce_loss + self.kl_weight * kl_loss
        
        return total_loss

def focal_loss(pred, gt, alpha=2, beta=4, weights=None):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, beta)
    loss = 0
    pos_loss = torch.log(pred + 1e-8) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-8) * torch.pow(pred, alpha) * neg_weights * neg_inds
    if weights is not None:
        pos_loss *= weights
        neg_loss *= weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss

