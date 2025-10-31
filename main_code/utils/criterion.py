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

class MV_Softmax(nn.Module):
    """
    Mis-classified Vector Guided Softmax Loss (MV-Softmax)
    Paper: https://arxiv.org/abs/1912.00829

    Fully aligned with your SphereFace / CosFace / ArcFace / CurricularFace heads.
    """
    def __init__(self,
                 feat_dim: int,
                 num_class: int,
                 margin: float = 0.35,
                 mv_weight: float = 1.12,
                 s: float = 32.0,
                 margin_type: str = 'arc',   # 'am' or 'arc'
                 device_id=None):
        """
        Args:
            feat_dim     : dimension of the backbone embedding
            num_class    : number of identities
            margin       : angular margin (m)
            mv_weight    : λ in the paper (hard-example scaling)
            s            : logit scale (same as Cos/Arc)
            margin_type  : 'am'  → additive margin (CosFace style)
                           'arc' → additive *angular* margin (ArcFace style)
            device_id    : (optional) list of GPUs for model-parallel (rare)
        """
        super().__init__()
        self.feat_dim   = feat_dim
        self.num_class  = num_class
        self.margin     = margin
        self.mv_weight  = mv_weight
        self.s          = s
        self.margin_type = margin_type.lower()
        self.device_id   = device_id

        assert self.margin_type in ('am', 'arc'), "margin_type must be 'am' or 'arc'"

        # ----- class prototypes (same shape as ArcFace) -----
        self.weight = nn.Parameter(torch.empty(num_class, feat_dim))
        # InsightFace-style init (large values → stable training)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        # pre-compute for Arc margin
        if self.margin_type == 'arc':
            self.cos_m = math.cos(margin)
            self.sin_m = math.sin(margin)
            self.th    = math.cos(math.pi - margin)      # threshold for easy-margin
            self.mm    = self.sin_m * margin

        print(f"MV_Softmax → margin={margin:.3f}, mv_weight={mv_weight:.2f}, "
              f"s={s:.1f}, type={self.margin_type.upper()}")

    # --------------------------------------------------------------------
    def get_proxy(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Return raw class centers for the supplied labels.
        Shape: [feat_dim, batch_size]  (same as all other heads)
        """
        return self.weight[:, labels].clone().detach()   # [D, N]

    # --------------------------------------------------------------------
    def forward(self, x: torch.Tensor, label: torch.Tensor):
        """
        Args:
            x     : raw backbone features, [N, feat_dim]
            label : ground-truth indices,   [N]

        Returns:
            [pre_margin_logits, final_logits], norms, loss_g, one_hot
            (identical to CosFace/ArcFace etc.)
        """
        with autocast(device_type='cuda'):
            # ---- feature norm (for MagFace / logging) ----
            norms = torch.norm(x, p=2, dim=1, keepdim=True)          # [N,1]

            # ---- L2-normalise both sides (ArcFace style) ----
            x_norm = F.normalize(x, dim=1)                           # [N,D]
            w_norm = F.normalize(self.weight, dim=1)                 # [C,D]

            # ---- cosine similarity ----
            if self.device_id is None:
                cos_theta = F.linear(x_norm, w_norm)                 # [N,C]
            else:
                # model-parallel (kept for completeness, rarely used)
                cos_theta = self._model_parallel_cos(x_norm, w_norm)

            cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
            cos_theta_cp = cos_theta.clone()                         # pre-margin copy

            # ---- target cosine ----
            target_cos = cos_theta[torch.arange(cos_theta.size(0)), label].view(-1, 1)

            # ----- margin handling ------------------------------------------------
            if self.margin_type == 'am':                     # Additive Margin (CosFace)
                final_target = torch.where(target_cos > self.margin,
                                           target_cos - self.margin,
                                           target_cos)
                mask = cos_theta > (target_cos - self.margin)

            else:                                            # Additive Angular Margin (ArcFace)
                sin_theta   = torch.sqrt(1.0 - target_cos**2 + 1e-9)
                cos_theta_m = target_cos * self.cos_m - sin_theta * self.sin_m
                final_target = torch.where(target_cos > 0.0, cos_theta_m, target_cos)
                mask = cos_theta > cos_theta_m

            # ----- MV-guided hard-example scaling ---------------------------------
            hard_example = cos_theta[mask]
            if hard_example.numel() > 0:
                cos_theta[mask] = self.mv_weight * hard_example + (self.mv_weight - 1.0)

            # ----- replace target column ------------------------------------------------
            final_target = final_target.to(cos_theta.dtype)
            cos_theta.scatter_(1, label.view(-1, 1), final_target)

            # ----- scale to logits ------------------------------------------------------
            logits = cos_theta * self.s
            pre_margin_logits = cos_theta_cp * self.s

            # ----- one-hot (for debugging / proxy analysis) ---------------------------
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)

        return [pre_margin_logits, logits], norms, 0, one_hot

    # --------------------------------------------------------------------
    def _model_parallel_cos(self, x_norm, w_norm):
        """Rarely used – kept for API parity with SphereFace/ArcFace."""
        x = x_norm
        sub_weights = torch.chunk(w_norm, len(self.device_id), dim=0)
        out = []
        for i, dev in enumerate(self.device_id):
            xi = x.to(dev)
            wi = sub_weights[i].to(dev)
            out.append(F.linear(xi, wi))
        return torch.cat(out, dim=1).to(self.device_id[0])

class MV_SoftmaxNet(nn.Module):
    """
    Backbone → embedding → MV_Softmax head
    """
    def __init__(self, num_classes: int, backbone):
        super().__init__()
        self.backbone = get_backbone(backbone)

        self.mv_head = MV_Softmax(
            feat_dim=FEATURE_DIM,
            num_class=num_classes,
            margin=M_mv,          # define in your config
            mv_weight=WEIGHT_mv,
            s=S_mv,
            margin_type=MARGIN_TYPE_mv   # 'am' or 'arc'
        )
        self.loss_model = f"mv_softmax_{MARGIN_TYPE_mv}"
        print(f"Initialize MV_Softmax model with backbone {backbone}")

    def forward(self, x, labels=None):
        feats = self.backbone(x)                 # [N, FEATURE_DIM]

        if self.training:
            assert labels is not None
            return self.mv_head(feats, labels)
        else:
            return feats

class AdaFace(nn.Module):
    """
    AdaFace: Quality Adaptive Margin for Face Recognition
    Paper: https://arxiv.org/abs/2105.08620
    
    Fully aligned with your SphereFace / CosFace / ArcFace / CurricularFace / MV_Softmax heads.
    """
    def __init__(self,
                 feat_dim: int,
                 num_class: int,
                 m: float = 0.4,
                 h: float = 0.333,
                 s: float = 64.0,
                 t_alpha: float = 1.0,
                 device_id=None):
        """
        Args:
            feat_dim  : dimension of backbone embedding
            num_class : number of identities
            m         : base margin
            h         : adaptive margin scale (0.333 → 66% samples in [-0.333, 0.333])
            s         : logit scale
            t_alpha   : EMA momentum for batch_mean/std
            device_id : (optional) GPUs for model-parallel
        """
        super().__init__()
        self.feat_dim   = feat_dim
        self.num_class  = num_class
        self.m          = m
        self.h          = h
        self.s          = s
        self.t_alpha    = t_alpha
        self.device_id  = device_id
        self.eps        = 1e-3

        # ---- class prototypes (same as CosFace: [D, C]) ----
        self.kernel = nn.Parameter(torch.empty(feat_dim, num_class))
        # InsightFace init: large values → stable training
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        # ---- EMA buffers ----
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)

        print(f"AdaFace → m={m:.3f}, h={h:.3f}, s={s:.1f}, t_alpha={t_alpha:.3f}")

    # --------------------------------------------------------------------
    def get_proxy(self, labels: torch.Tensor) -> torch.Tensor:
        """Return raw class centers. Shape: [feat_dim, batch_size]"""
        return self.kernel[:, labels].clone().detach()   # [D, N]

    # --------------------------------------------------------------------
    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            feats   : [N, feat_dim]  raw backbone features
            labels  : [N]            class indices

        Returns:
            [pre_margin_logits, final_logits], norms, loss_g, one_hot
            (identical to all other heads)
        """
        with autocast(device_type='cuda'):
            # ---- feature norms ----
            norms = torch.norm(feats, p=2, dim=1, keepdim=True)          # [N,1]

            # ---- L2-normalise both sides ----
            feats_norm = F.normalize(feats, dim=1)                       # [N,D]
            weight_norm = F.normalize(self.kernel, dim=0)                # [D,C]

            # ---- cosine similarity ----
            if self.device_id is None:
                cosine = torch.mm(feats_norm, weight_norm)               # [N,C]
            else:
                cosine = self._model_parallel_cos(feats_norm, weight_norm)

            cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)
            cosine_cp = cosine.clone()                                   # pre-margin

            # ---- adaptive margin scaler (EMA on norms) -----------------
            safe_norms = torch.clamp(norms, min=0.001, max=100).detach()

            with torch.no_grad():
                mean = safe_norms.mean()
                std = safe_norms.std()
                self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
                self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

            margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
            margin_scaler = torch.clamp(margin_scaler * self.h, -1, 1)   # [-h, h]

            # ---- one-hot ----
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)

            # ---- AdaFace margins (adaptive arc + additive) ------------
            # 1. Angular margin: cos(θ + m * margin_scaler)
            theta = cosine.acos()
            m_arc = one_hot * (self.m * margin_scaler * -1)              # negative for arc
            theta_m = torch.clamp(theta + m_arc, self.eps, math.pi - self.eps)
            cosine_m_arc = theta_m.cos()

            # 2. Additive margin: -m * (1 + margin_scaler)
            g_add = self.m + (self.m * margin_scaler)
            m_cos = one_hot * g_add
            cosine = cosine_m_arc - m_cos

            # ---- scale to logits ----
            logits = cosine * self.s
            pre_margin_logits = cosine_cp * self.s

        return [pre_margin_logits, logits], norms, 0, one_hot

    # --------------------------------------------------------------------
    def _model_parallel_cos(self, feats_norm, weight_norm):
        """Model-parallel cosine (rarely used – API parity)."""
        sub_weights = torch.chunk(weight_norm, len(self.device_id), dim=1)
        out = []
        for i, dev in enumerate(self.device_id):
            xi = feats_norm.to(dev)
            wi = sub_weights[i].to(dev)
            out.append(torch.mm(xi, wi))
        return torch.cat(out, dim=1).to(self.device_id[0])

class AdaFaceNet(nn.Module):
    """
    Backbone → embedding → AdaFace head
    """
    def __init__(self, num_classes: int, backbone):
        super().__init__()
        # ----- backbone -----
        self.backbone = get_backbone(backbone)

        # ----- AdaFace head -----
        self.adaface = AdaFace(
            feat_dim=FEATURE_DIM,
            num_class=num_classes,
            m=M_ada,
            h=H_ada,
            s=S_ada,
            t_alpha=T_ALPHA_ada
        )
        self.loss_model = "adaface"
        print(f"Initialize AdaFace model with backbone {backbone}")

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        feats = self.backbone(x)                 # [N, FEATURE_DIM]

        if self.training:
            assert labels is not None
            return self.adaface(feats, labels)   # → [pre*s, logits], norms, 0, one_hot
        else:
            return feats

class ElasticCosFace(nn.Module):
    """
    Elastic CosFace: Elastic Additive Margin
    """
    def __init__(self,
                 feat_dim: int,
                 num_class: int,
                 s: float = 64.0,
                 m: float = 0.35,
                 std: float = 0.0125,
                 plus: bool = False,
                 device_id=None):
        super().__init__()
        self.feat_dim   = feat_dim
        self.num_class  = num_class
        self.s          = s
        self.m          = m
        self.std        = std
        self.plus       = plus
        self.device_id  = device_id

        self.kernel = nn.Parameter(torch.empty(feat_dim, num_class))
        nn.init.normal_(self.kernel, std=0.01)

        print(f"ElasticCosFace → s={s:.1f}, m={m:.3f}, std={std:.4f}, plus={plus}")

    # --------------------------------------------------------------------
    def get_proxy(self, labels: torch.Tensor) -> torch.Tensor:
        return self.kernel[:, labels].clone().detach()

    # --------------------------------------------------------------------
    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        with autocast(device_type='cuda'):
            norms = torch.norm(feats, p=2, dim=1, keepdim=True)

            feats_norm = F.normalize(feats, dim=1)
            weight_norm = F.normalize(self.kernel, dim=0)

            if self.device_id is None:
                cos_theta = torch.mm(feats_norm, weight_norm)
            else:
                cos_theta = self._model_parallel_cos(feats_norm, weight_norm)

            cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
            cos_theta_cp = cos_theta.clone()

            index = torch.where(labels != -1)[0]
            if index.numel() == 0:
                logits = cos_theta * self.s
                one_hot = torch.zeros_like(logits)
                return [cos_theta_cp * self.s, logits], norms, 0, one_hot

            margin = torch.normal(mean=self.m, std=self.std,
                                  size=(index.size(0), 1), device=cos_theta.device)
            margin = margin.clamp(self.m - self.std, self.m + self.std)

            if self.plus:
                with torch.no_grad():
                    target_cos = cos_theta[index, labels[index]]
                    _, rank = torch.sort(target_cos, descending=True)
                    margin, _ = torch.sort(margin.squeeze(1))
                    margin = margin[rank].unsqueeze(1)

            cos_theta[index, labels[index]] -= margin.squeeze(1)
            logits = cos_theta * self.s
            pre_margin_logits = cos_theta_cp * self.s

            one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        return [pre_margin_logits, logits], norms, 0, one_hot

    def _model_parallel_cos(self, x_norm, w_norm):
        sub_weights = torch.chunk(w_norm, len(self.device_id), dim=1)
        out = []
        for i, dev in enumerate(self.device_id):
            xi = x_norm.to(dev)
            wi = sub_weights[i].to(dev)
            out.append(torch.mm(xi, wi))
        return torch.cat(out, dim=1).to(self.device_id[0])

class ElasticCosFaceNet(nn.Module):
    def __init__(self, num_classes: int, backbone):
        super().__init__()
        self.backbone = get_backbone(backbone)
        self.head = ElasticCosFace(
            feat_dim=FEATURE_DIM,
            num_class=num_classes,
            s=S_elastic_cos,
            m=M_elastic_cos,
            std=STD_elastic_cos,
            plus=PLUS_elastic_cos
        )
        self.loss_model = "elastic_cosface"
        print(f"Initialize ElasticCosFace model with backbone {backbone}")

    def forward(self, x, labels=None):
        feats = self.backbone(x)
        if self.training:
            assert labels is not None
            return self.head(feats, labels)
        return feats

class ElasticArcFace(nn.Module):
    """
    Elastic ArcFace: Elastic Margin for Face Recognition
    Paper: https://arxiv.org/abs/2004.03096

    Fully aligned with your existing heads.
    """
    def __init__(self,
                 feat_dim: int,
                 num_class: int,
                 s: float = 64.0,
                 m: float = 0.50,
                 std: float = 0.0125,
                 plus: bool = False,
                 device_id=None):
        super().__init__()
        self.feat_dim   = feat_dim
        self.num_class  = num_class
        self.s          = s
        self.m          = m
        self.std        = std
        self.plus       = plus
        self.device_id  = device_id

        # ---- class prototypes: [D, C] (same as CosFace, CurricularFace) ----
        self.kernel = nn.Parameter(torch.empty(feat_dim, num_class))
        nn.init.normal_(self.kernel, std=0.01)

        print(f"ElasticArcFace → s={s:.1f}, m={m:.3f}, std={std:.4f}, plus={plus}")

    # --------------------------------------------------------------------
    def get_proxy(self, labels: torch.Tensor) -> torch.Tensor:
        return self.kernel[:, labels].clone().detach()   # [D, N]

    # --------------------------------------------------------------------
    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        with autocast(device_type='cuda'):
            # ---- feature norms (for MagFace / logging) ----
            norms = torch.norm(feats, p=2, dim=1, keepdim=True)          # [N,1]

            # ---- L2-normalize ----
            feats_norm = F.normalize(feats, dim=1)                       # [N,D]
            weight_norm = F.normalize(self.kernel, dim=0)                # [D,C]

            # ---- cosine similarity ----
            if self.device_id is None:
                cos_theta = torch.mm(feats_norm, weight_norm)            # [N,C]
            else:
                cos_theta = self._model_parallel_cos(feats_norm, weight_norm)

            cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
            cos_theta_cp = cos_theta.clone()                             # pre-margin

            # ---- valid indices (ignore label == -1) ----
            index = torch.where(labels != -1)[0]
            if index.numel() == 0:
                # No valid labels → return unmodified
                logits = cos_theta * self.s
                one_hot = torch.zeros_like(logits)
                return [cos_theta_cp * self.s, logits], norms, 0, one_hot

            # ---- sample elastic margin per target ----
            margin = torch.normal(mean=self.m, std=self.std,
                                  size=(index.size(0), 1), device=cos_theta.device)
            margin = margin.clamp(self.m - self.std, self.m + self.std)   # optional

            if self.plus:
                # Rank-based assignment: hardest gets largest margin
                with torch.no_grad():
                    target_cos = cos_theta[index, labels[index]]
                    _, rank = torch.sort(target_cos, descending=True)
                    margin, _ = torch.sort(margin.squeeze(1))
                    margin = margin[rank].unsqueeze(1)

            # ---- apply margin: cos(θ + m) → θ + m → cos ----
            target_theta = cos_theta[index, labels[index]].acos()
            theta_m = target_theta + margin.squeeze(1)
            theta_m = torch.clamp(theta_m, 0, math.pi)
            cos_theta_m = theta_m.cos().to(cos_theta.dtype)

            # ---- replace target logits ----
            cos_theta[index, labels[index]] = cos_theta_m

            # ---- scale ----
            logits = cos_theta * self.s
            pre_margin_logits = cos_theta_cp * self.s

            # ---- one-hot (full batch) ----
            one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        return [pre_margin_logits, logits], norms, 0, one_hot

    def _model_parallel_cos(self, x_norm, w_norm):
        sub_weights = torch.chunk(w_norm, len(self.device_id), dim=1)
        out = []
        for i, dev in enumerate(self.device_id):
            xi = x_norm.to(dev)
            wi = sub_weights[i].to(dev)
            out.append(torch.mm(xi, wi))
        return torch.cat(out, dim=1).to(self.device_id[0])

class ElasticArcFaceNet(nn.Module):
    def __init__(self, num_classes: int, backbone):
        super().__init__()
        self.backbone = get_backbone(backbone)
        self.head = ElasticArcFace(
            feat_dim=FEATURE_DIM,
            num_class=num_classes,
            s=S_elastic_arc,
            m=M_elastic_arc,
            std=STD_elastic_arc,
            plus=PLUS_elastic_arc
        )
        self.loss_model = "elastic_arcface"
        print(f"Initialize ElasticArcFace model with backbone {backbone}")

    def forward(self, x, labels=None):
        feats = self.backbone(x)
        if self.training:
            assert labels is not None
            return self.head(feats, labels)
        return feats

