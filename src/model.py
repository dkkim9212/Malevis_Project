import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_model(num_classes):

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model