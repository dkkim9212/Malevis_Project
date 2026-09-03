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
# Stage 1
# FC Layer만 학습
# ==========================================

def freeze_backbone(model):

    # 전체 파라미터 동결
    for param in model.parameters():
        param.requires_grad = False

    # FC Layer만 학습 가능
    for param in model.fc.parameters():
        param.requires_grad = True


# ==========================================
# Stage 2
# Layer4 + FC 학습
# ==========================================

def unfreeze_layer4(model):

    # 일단 전체 동결
    for param in model.parameters():
        param.requires_grad = False

    # ResNet의 마지막 블록 해제
    for param in model.layer4.parameters():
        param.requires_grad = True

    # FC Layer 해제
    for param in model.fc.parameters():
        param.requires_grad = True