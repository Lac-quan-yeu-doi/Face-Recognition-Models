from torchvision.models import (
    resnet18, resnet50,
    ResNet18_Weights, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
import torch.nn as nn

from .config import FEATURE_DIM

def get_backbone(backbone_name='resnet18'):
    if backbone_name == 'resnet18':
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, FEATURE_DIM)

    elif backbone_name == 'resnet50':
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, FEATURE_DIM)

    elif backbone_name == 'efficientnet_b0':
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, FEATURE_DIM)

    elif backbone_name == 'mobilenet_v2':
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, FEATURE_DIM)

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return backbone
