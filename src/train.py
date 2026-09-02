import torch
import torch.nn as nn
import argparse

from dataset import get_dataloaders
from model import get_model


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    type=str,
    required=True
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32
)

parser.add_argument(
    "--epochs",
    type=int,
    default=5
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.0001
)

args = parser.parse_args()


# ==============================
# Device
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치 :", device)


# ==============================
# Dataset
# ==============================

train_loader, val_loader, classes = get_dataloaders(
    args.data_dir,
    args.batch_size
)

print("클래스 수 :", len(classes))
print("Train 이미지 :", len(train_loader.dataset))
print("Validation 이미지 :", len(val_loader.dataset))


# ==============================
# Model
# ==============================

model = get_model(
    num_classes=len(classes)
)

model = model.to(device)


# ==============================
# Loss
# ==============================

criterion = nn.CrossEntropyLoss()


# ==============================
# Optimizer
# ==============================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)


# ==============================
# Best Model
# ==============================

best_val_accuracy = 0.0


# ==============================
# Training
# ==============================

for epoch in range(args.epochs):

    # --------------------------
    # Train
    # --------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * train_correct / train_total
    )


    # --------------------------
    # Validation
    # --------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    # --------------------------
    # 결과 출력
    # --------------------------

    print(
        f"\nEpoch [{epoch + 1}/{args.epochs}]"
    )

    print(
        f"Train Loss: "
        f"{train_loss / len(train_loader):.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: "
        f"{val_loss / len(val_loader):.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    # --------------------------
    # Best Model 저장
    # --------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            "Best Model 저장!"
        )


print("\n학습 완료")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)