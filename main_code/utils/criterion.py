import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torchvision.models import resnet50, resnet18, ResNet18_Weights, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v2, MobileNet_V2_Weights

from .config import *
from .backbones import get_backbone

### Reimplement
class SphereFace(nn.Module):
    """
    SphereFace: Deep Hypersphere Embedding for Face Recognition
    Paper: https://arxiv.org/abs/1704.08063
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device_id=None,
                 m: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.device_id = device_id

        # ---- Annealing parameters ----
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0

        # ---- Class prototypes ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # ---- Chebyshev polynomials for cos(mθ) ----
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        print(f"init SphereFace → m={self.m}, base={self.base}, gamma={self.gamma}")

    # --------------------------------------------------------------------
    def get_proxy(self, labels: torch.Tensor) -> torch.Tensor:
        """Return raw class centers for given labels."""
        return self.weight.permute(1, 0)[:, labels].clone().detach()

    # --------------------------------------------------------------------
    def forward(self, input: torch.Tensor, label: torch.Tensor):
        self.iter += 1
        # Annealing: λ decreases over time
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-self.power))

        # ---- Cosine similarity (normalized) ----
        with autocast(device_type='cuda'):
            if self.device_id is None:
                cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
            else:
                # Model parallel (rarely used)
                x = input
                sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
                temp_x = x.cuda(self.device_id[0])
                weight = sub_weights[0].cuda(self.device_id[0])
                cos_theta = F.linear(F.normalize(temp_x), F.normalize(weight))
                for i in range(1, len(self.device_id)):
                    temp_x = x.cuda(self.device_id[i])
                    weight = sub_weights[i].cuda(self.device_id[i])
                    cos_theta = torch.cat((
                        cos_theta,
                        F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])
                    ), dim=1)

            cos_theta = cos_theta.clamp(-1, 1)
            cos_theta_cp = cos_theta.clone()

            # ---- cos(mθ) using Chebyshev ----
            cos_m_theta = self.mlambda[self.m](cos_theta)

            # ---- Compute θ and k ----
            theta = cos_theta.data.acos()  # [N, C]
            k = (self.m * theta / math.pi).floor()

            # ---- phi(θ) = (-1)^k * cos(mθ) - 2k ----
            phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k

            # ---- Feature norms ----
            NormOfFeature = torch.norm(input, p=2, dim=1, keepdim=True)  # [N, 1]

            # ---- One-hot ----
            one_hot = torch.zeros_like(cos_theta)
            if self.device_id is not None:
                one_hot = one_hot.cuda(self.device_id[0])
            one_hot.scatter_(1, label.view(-1, 1), 1)

            # ---- Final output (with annealing) ----
            output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
            output = output * NormOfFeature

        return [cos_theta_cp * NormOfFeature, output], NormOfFeature, 0, one_hot

class SphereFaceNet(nn.Module):
    """
    Backbone → embedding → SphereFace head
    """
    def __init__(self, num_classes: int, backbone):
        super().__init__()
        # ----- backbone (feel free to swap) -----
        self.backbone = get_backbone(backbone)

        # ----- SphereFace head -----
        self.sphereface = SphereFace(
            in_features=FEATURE_DIM,
            out_features=num_classes,
            m=M_sphere
        )
        self.loss_model = "sphereface"   # for logging / compatibility
        print(f"Initialize SphereFace model with backbone {backbone}")

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        feats = self.backbone(x)                 # [N, FEATURE_DIM]

        if self.training:
            assert labels is not None
            return self.sphereface(feats, labels)   # → [cos*||x||, logits], norms, loss_g, one_hot
        else:
            return feats

class CosFace(nn.Module):
    """
    CosFace (Additive Angular Margin) – InsightFace style
    """
    def __init__(self, embedding_size=512, classnum=51332,
                 s: float = 64.0, m: float = 0.4):
        super().__init__()
        self.classnum = classnum
        self.s = s
        self.m = m
        self.eps = 1e-4

        # ---- weight (kernel) -------------------------------------------------
        self.kernel = nn.Parameter(torch.empty(embedding_size, classnum))
        # InsightFace init: large values → stable training
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        print(f"init CosFace → s={self.s:.2f}, m={self.m:.3f}")

    # --------------------------------------------------------------------
    def get_proxy(self, labels: torch.Tensor) -> torch.Tensor:
        """Return the proxy vectors for the given labels (used for analysis)."""
        return self.kernel[:, labels].clone().detach()

    # --------------------------------------------------------------------
    def forward(self, embeddings: torch.Tensor, label: torch.Tensor):
        """
        Args:
            embeddings: [N, embed_dim]  (raw features from backbone)
            label:      [N]             (class indices)

        Returns:
            [cosine*s, logits], norms, loss_g, one_hot
        """
        with autocast(device_type='cuda'):
            # L2-normalise both sides
            embeds_norm = F.normalize(embeddings, dim=1)          # [N, D]
            weight_norm = F.normalize(self.kernel, dim=0)         # [D, C]

            cosine = torch.mm(embeds_norm, weight_norm)           # [N, C]
            cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)    # stability

            cosine_cp = cosine.clone()

            # one-hot mask for target class
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)

            # ----- additive margin only on target class -----
            cosine = cosine - one_hot * self.m

            # final logits
            logits = cosine * self.s

            # feature norm (for MagFace / auxiliary loss)
            norms = torch.norm(embeddings, dim=1, keepdim=True)

            # loss_g = 0  (placeholder – set >0 in MagFace)
            loss_g = 0

        return [cosine_cp * self.s, logits], norms, loss_g, one_hot

class CosFaceNet(nn.Module):
    """
    Backbone → embedding → CosFace head
    """
    def __init__(self, num_classes: int, backbone):
        super().__init__()
        # ----- backbone (feel free to swap) -----
        self.backbone = get_backbone(backbone)

        # ----- CosFace head -----
        self.cosface = CosFace(
            embedding_size=FEATURE_DIM,
            classnum=num_classes,
            s=S_cos,
            m=M_cos
        )
        self.loss_model = "cosface"          # for compatibility with your train loop
        print(f"Initialize CosFace model with backbone {backbone}")

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Training:  returns the full tuple expected by train_model
        Inference: returns raw embeddings
        """
        feats = self.backbone(x)                # [N, FEATURE_DIM]

        if self.training:
            assert labels is not None
            return self.cosface(feats, labels)  # → [cosine*s, logits], norms, loss_g, one_hot
        else:
            return feats

class ArcFace(nn.Module):
    r"""Correct ArcFace Implementation (matches your original)"""
    def __init__(self, embed_size, num_classes, device_id=None, s=64.0, m=0.50, easy_margin=True):
        super(ArcFace, self).__init__()
        self.in_features = embed_size
        self.out_features = num_classes
        self.device_id = device_id
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    # --------------------------------------------------------------
    def get_proxy(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Return the **raw** class-centers (proxies) for the supplied labels.
        Shape: [embed_dim, batch_size]  (same as CosFace.get_proxy)
        """
        # self.weight: [C, D]  →  pick columns → [D, N]
        return self.weight[:, labels].clone().detach()

    def forward(self, input, label):
        with autocast(device_type='cuda'):
            # L2 normalize input and weight
            input_norm = F.normalize(input)
            weight_norm = F.normalize(self.weight)

            if self.device_id is None:
                cosine = F.linear(input_norm, weight_norm)  # [N, C]
            else:
                # Model parallel support (chunk weights across GPUs)
                x = input
                sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
                temp_x = x.cuda(self.device_id[0])
                weight = sub_weights[0].cuda(self.device_id[0])
                cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
                for i in range(1, len(self.device_id)):
                    temp_x = x.cuda(self.device_id[i])
                    weight = sub_weights[i].cuda(self.device_id[i])
                    cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)

            cosine_cp = cosine.clone()
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-9, 1.0))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            # One-hot encoding
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)

            # Combine: target gets phi, others get cosine
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s  # Scale

            # Feature norm (for potential MagFace-style loss)
            norms = torch.norm(input, dim=1, keepdim=True)
            loss_g = 0  # Placeholder — set in MagFace or similar

        return [cosine_cp * self.s, output], norms, loss_g, one_hot

class ArcFaceNet(nn.Module):
    def __init__(self, num_classes, backbone):
        super(ArcFaceNet, self).__init__()
        self.backbone = get_backbone(backbone)

        self.arcface = ArcFace(
            embed_size=FEATURE_DIM,
            num_classes=num_classes,
            s=S_arc,
            m=M_arc,
            easy_margin=True
        )
        self.loss_model = "arcface"  # For compatibility with do_train
        print(f"Initialize ArcFace model with backbone {backbone}")

    def forward(self, x, labels=None):

        features = self.backbone(x)
        if self.training:
            assert labels is not None
            output = self.arcface(features, labels)
            return output  # [cosine*s, logits], norms, loss_g, one_hot
        return features

class CurricularFace(nn.Module):
    """
    CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition
    Paper: https://arxiv.org/abs/2004.00288
    """
    def __init__(self,
                 feat_dim: int,
                 num_class: int,
                 m: float = 0.5,
                 s: float = 64.0,
                 momentum: float = 0.01):
        super().__init__()
        self.m = m
        self.s = s
        self.momentum = momentum

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)      # thres
        self.mm = math.sin(math.pi - m) * m

        # ---- class prototypes (kernel) ----
        self.kernel = nn.Parameter(torch.empty(feat_dim, num_class))
        nn.init.normal_(self.kernel, std=0.01)

        # EMA of target cosine (curriculum difficulty)
        self.register_buffer('t', torch.zeros(1))

        print(f"init CurricularFace → s={self.s:.2f}, m={self.m:.3f}, momentum={self.momentum:.3f}")

    # --------------------------------------------------------------------
    def get_proxy(self, labels: torch.Tensor) -> torch.Tensor:
        """Return raw class centers for given labels."""
        return self.kernel[:, labels].clone().detach()

    # --------------------------------------------------------------------
    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            feats:  [N, feat_dim]  (raw backbone features)
            labels: [N]            (class indices)

        Returns:
            [origin_cos*s, logits], norms, loss_g, one_hot
        """
        with autocast(device_type='cuda'):
            # ---- feature norms (for MagFace / logging) ----
            norms = torch.norm(feats, dim=1, keepdim=True)          # [N, 1]

            # ---- L2-normalise both sides ----
            kernel_norm = F.normalize(self.kernel, dim=0)           # [D, C]
            feats_norm  = F.normalize(feats, dim=1)                 # [N, D]

            # ---- cosine similarity ----
            cos_theta = torch.mm(feats_norm, kernel_norm)           # [N, C]
            cos_theta = cos_theta.clamp(-1, 1)

            # keep a copy for accuracy (pre-margin)
            origin_cos = cos_theta.clone()

            # ---- target logits ----
            target_logit = cos_theta[torch.arange(feats.size(0)), labels].view(-1, 1)

            # cos(θ + m)
            sin_theta   = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m

            # mask: non-target cosine > cos(θ + m)
            mask = cos_theta > cos_theta_m

            # final target logit (only apply full margin if target is hard enough)
            final_target_logit = torch.where(
                target_logit > self.threshold,
                cos_theta_m,
                target_logit - self.mm
            ).to(cos_theta.dtype)

            # ---- curriculum: adaptive scaling of hard negatives ----
            hard_example = cos_theta[mask]
            with torch.no_grad():
                # EMA update: t = momentum * mean_target + (1-momentum) * t
                self.t = (target_logit.mean() * self.momentum +
                        (1 - self.momentum) * self.t).to(hard_example.dtype)

            cos_theta[mask] = hard_example * (self.t + hard_example)

            # replace target column
            cos_theta.scatter_(1, labels.view(-1, 1), final_target_logit)

            # final scaled logits
            logits = cos_theta * self.s

            # one-hot for analysis / debugging
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, labels.view(-1, 1), 1)

        return [origin_cos * self.s, logits], norms, 0, one_hot

class CurricularFaceNet(nn.Module):
    """
    Backbone → embedding → CurricularFace head
    """
    def __init__(self, num_classes: int, backbone):
        super().__init__()
        # ----- backbone (feel free to swap) -----
        self.backbone = get_backbone(backbone)

        # ----- CurricularFace head -----
        self.curricular = CurricularFace(
            feat_dim=FEATURE_DIM,
            num_class=num_classes,
            m=M_curricular,
            s=S_curricular,
            momentum=MOMENTUM_curricular
        )
        self.loss_model = "curricularface"   # for logging / compatibility
        print(f"Initialize CurricularFace model with backbone {backbone}")

    # ----------------------------------------------------------------
    def forward(self, x, labels=None):
        feats = self.backbone(x)                 # [N, FEATURE_DIM]

        if self.training:
            assert labels is not None
            return self.curricular(feats, labels)   # → [origin*s, logits], norms, loss_g, one_hot
        else:
            return feats












class MagFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, l_margin=0.45, u_margin=0.8, l_a=10.0, u_a=110.0, lambda_g=35.0):
        super(MagFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s  # Scale: Typically 64.0
        self.l_margin = l_margin  # Lower margin bound
        self.u_margin = u_margin  # Upper margin bound
        self.l_a = l_a  # Lower magnitude bound
        self.u_a = u_a  # Upper magnitude bound
        self.lambda_g = lambda_g  # Regularizer weight
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        print(f"Initialize MagFace with scale {self.s}, l_margin {self.l_margin}, u_margin {self.u_margin}, l_a {self.l_a}, u_a {self.u_a}, lambda_g {self.lambda_g}")

    def forward(self, x, labels):
        with autocast(device_type='cuda'):
            w = nn.functional.normalize(self.weight, p=2, dim=1)
            norm_x = x.norm(p=2, dim=1, keepdim=True)
            a = torch.clamp(norm_x, min=self.l_a, max=self.u_a)
            cos_theta = torch.mm(x, w.t()) / norm_x  # cos_theta = (x · w) / ||x|| since ||w||=1
            batch_size = x.size(0)
            target_cos_theta = cos_theta[torch.arange(batch_size), labels].view(-1, 1)
            m_a = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (a - self.l_a) + self.l_margin
            cos_m = torch.cos(m_a)
            sin_m = torch.sin(m_a)
            sin_theta = torch.sqrt((1.0 - torch.pow(target_cos_theta, 2)).clamp(0, 1))
            cos_theta_m = target_cos_theta * cos_m - sin_theta * sin_m
            thres = torch.cos(torch.pi - m_a)
            mm = torch.sin(torch.pi - m_a) * m_a
            final_target_logit = torch.where(target_cos_theta > thres, cos_theta_m, target_cos_theta - mm)
            cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit.to(cos_theta.dtype))
            scaled_logits = cos_theta * self.s
            loss_ce = nn.CrossEntropyLoss()(scaled_logits, labels)
            g = 1 / a + a / (self.u_a ** 2)
            loss_g = self.lambda_g * g.mean()
            loss = loss_ce + loss_g
        return loss

class MagFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(MagFaceNet, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, FEATURE_DIM)
        self.magface = MagFaceLoss(in_features=FEATURE_DIM, num_classes=num_classes, s=S_mag, l_margin=M_l_mag, u_margin=M_u_mag, l_a=A_l_mag, u_a=A_u_mag, lambda_g=LAMBDA_g_mag)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        if self.training:
            loss = self.magface(features, labels)
            return loss, features
        return features

