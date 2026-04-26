import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


def build_model():
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # 🔥 Unfreeze last dense block
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True

    # Replace classifier
    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(128, 1)  # NO sigmoid
    )

    return model