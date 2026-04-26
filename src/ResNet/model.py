import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # 🔥 Unfreeze last layers (important)
    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(128, 1)   # NO sigmoid
    )

    return model