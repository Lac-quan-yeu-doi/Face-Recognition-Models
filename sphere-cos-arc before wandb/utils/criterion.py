import math
import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision.models import resnet50, resnet18, ResNet18_Weights, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v2, MobileNet_V2_Weights

from .config import *

class AngularSoftmaxLoss(nn.Module):
    def __init__(self, in_features, num_classes, m=4, s=30.0):
        super(AngularSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.m = m  # Note: Tune m (e.g., 2, 3, 4) and s (e.g., 10, 30) for better performance
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        print(f"Initialize SphereFace with margin {self.m}, scale {self.s}")

    def forward(self, x, labels):
        with autocast(device_type='cuda'):
            x = nn.functional.normalize(x, p=2, dim=1)
            w = nn.functional.normalize(self.weight, p=2, dim=1)
            cos_theta = torch.mm(x, w.t())
            batch_size = x.size(0)
            target_cos_theta = cos_theta[torch.arange(batch_size), labels]
            theta = torch.acos(target_cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            cos_m_theta = torch.cos(self.m * theta)
            cos_theta = self.s * cos_theta
            cos_m_theta = self.s * cos_m_theta
            cos_m_theta = cos_m_theta.to(dtype=cos_theta.dtype)
            cos_theta[torch.arange(batch_size), labels] = cos_m_theta
            loss = nn.CrossEntropyLoss()(cos_theta, labels)
        return loss

class SphereFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(SphereFaceNet, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512) 
        # self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512) 
        # self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 128)  
        # self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 128) 
        self.asoftmax = AngularSoftmaxLoss(in_features=512, num_classes=num_classes, m=M_sphere, s=S_sphere)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        if self.training:
            loss = self.asoftmax(features, labels)
            return loss, features
        return features

class CosFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, m=0.35, s=30.0):
        super(CosFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.m = m  # Margin: Tune m (e.g., 0.2, 0.35, 0.5) and s (e.g., 20, 30, 64) for better performance
        self.s = s  # Scale
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        print(f"Initialize CosFace with margin {self.m}, scale {self.s}")

    def forward(self, x, labels):
        with autocast(device_type='cuda'):
            x = nn.functional.normalize(x, p=2, dim=1)
            w = nn.functional.normalize(self.weight, p=2, dim=1)
            cos_theta = torch.mm(x, w.t())
            batch_size = x.size(0)
            target_cos_theta = cos_theta[torch.arange(batch_size), labels]
            # Apply margin to target cosine
            cos_theta_m = target_cos_theta - self.m
            cos_theta_m = cos_theta_m.to(dtype=cos_theta.dtype)
            cos_theta[torch.arange(batch_size), labels] = cos_theta_m
            # Scale the logits
            scaled_logits = self.s * cos_theta
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
        return loss

class CosFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(CosFaceNet, self).__init__()
        # self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512) 
        # self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 128)  
        # self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 128) 

        self.cosface = CosFaceLoss(in_features=512, num_classes=num_classes, m=M_cos, s=S_cos)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        if self.training:
            loss = self.cosface(features, labels)
            return loss, features
        return features

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, m=0.5, s=30.0):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.m = m  # Angular margin: Tune m (e.g., 0.3, 0.5, 0.7) and s (e.g., 30, 64) for better performance
        self.s = s  # Scale
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        print(f"Initialize ArcFace with margin {self.m}, scale {self.s}")

    def forward(self, x, labels):
        with autocast(device_type='cuda'):
            x = nn.functional.normalize(x, p=2, dim=1)
            w = nn.functional.normalize(self.weight, p=2, dim=1)
            cos_theta = torch.mm(x, w.t())
            batch_size = x.size(0)
            target_cos_theta = cos_theta[torch.arange(batch_size), labels]
            # Compute cos(θ + m) for target classes
            theta = torch.acos(target_cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            cos_theta_m = torch.cos(theta + self.m)
            cos_theta_m = cos_theta_m.to(dtype=cos_theta.dtype)
            cos_theta[torch.arange(batch_size), labels] = cos_theta_m
            # Scale the logits
            scaled_logits = self.s * cos_theta
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
        return loss

class ArcFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(ArcFaceNet, self).__init__()
        # self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512) 
        # self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 128)  
        # self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 128) 

        self.arcface = ArcFaceLoss(in_features=512, num_classes=num_classes, m=M_arc, s=S_arc)

        
    def forward(self, x, labels=None):
        features = self.backbone(x)
        if self.training:
            loss = self.arcface(features, labels)
            return loss, features
        return features

class CurricularFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, m=0.5, s=64.0, momentum=0.01):
        super(CurricularFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.m = m  # Margin: As per paper, typically 0.5
        self.s = s  # Scale: As per paper, typically 64.0
        self.momentum = momentum  # Momentum for new value in EMA, equivalent to (1 - α) where α=0.99 in paper
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.thres = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('t', torch.zeros(1))
        print(f"Initialize CurricularFace with margin {self.m}, scale {self.s}")

    def forward(self, x, labels):
        with autocast(device_type='cuda'):
            x = nn.functional.normalize(x, p=2, dim=1)
            w = nn.functional.normalize(self.weight, p=2, dim=1)
            cos_theta = torch.mm(x, w.t())
            batch_size = x.size(0)
            target_cos_theta = cos_theta[torch.arange(batch_size), labels].view(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(target_cos_theta, 2))
            cos_theta_m = target_cos_theta * self.cos_m - sin_theta * self.sin_m
            mask = cos_theta > cos_theta_m
            final_target_logit = torch.where(target_cos_theta > self.thres, cos_theta_m, target_cos_theta - self.mm)
            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_cos_theta.mean() * self.momentum + (1 - self.momentum) * self.t
            cos_theta[mask] = hard_example * (self.t + hard_example)
            cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
            scaled_logits = cos_theta * self.s
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
        return loss

class CurricularFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(CurricularFaceNet, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)
        self.curricularface = CurricularFaceLoss(in_features=512, num_classes=num_classes, m=M_curricular, s=S_curricular, momentum=MOMENTUM_curricular)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        if self.training:
            loss = self.curricularface(features, labels)
            return loss, features
        return features
