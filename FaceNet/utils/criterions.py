import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()

class CosFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, m=0.35, s=64):
        super().__init__()
        self.m = m  # cosine margin (float)
        self.s = s
        self.W = nn.Parameter(torch.randn(feat_dim, num_classes))

        nn.init.xavier_uniform_(self.W)

    def forward(self, x, labels):
        x_norm = F.normalize(x, dim=1)
        W_norm = F.normalize(self.W, dim=0)

        cos_theta = torch.matmul(x_norm, W_norm).clamp(-1.0, 1.0)
        one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).float()

        # Apply margin only to target class
        logits = self.s * (cos_theta - one_hot * self.m)
        loss = F.cross_entropy(logits, labels)
        return loss

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, m=0.5, s=64):
        super().__init__()
        self.m = m  # angular margin (in radians)
        self.s = s
        self.W = nn.Parameter(torch.randn(feat_dim, num_classes))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, labels):
        x_norm = F.normalize(x, dim=1)
        W_norm = F.normalize(self.W, dim=0)

        cos_theta = torch.matmul(x_norm, W_norm).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        cos_m_theta = torch.cos(theta + self.m)

        one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).float()
        logits = self.s * (one_hot * cos_m_theta + (1 - one_hot) * cos_theta)
        loss = F.cross_entropy(logits, labels)
        return loss
