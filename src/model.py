import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


def get_model(num_classes):

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model


# ==========================================
# Warm-up
# FC Layer만 학습
# ==========================================

def freeze_backbone(model):

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


# ==========================================
# Fine-tuning
# Layer3 + Layer4 + FC 학습
# ==========================================

def unfreeze_layer3_layer4(model):

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True
