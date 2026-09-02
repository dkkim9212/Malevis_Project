import torch
import os
import torch.nn as nn
import argparse
import time

from tqdm.auto import tqdm

from dataset import get_dataloaders
from model import get_model


# ==========================================
# 실행 옵션
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--save_dir",
    type=str,
    default="./models"
)

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

os.makedirs(
    args.save_dir,
    exist_ok=True
)

best_model_path = os.path.join(
    args.save_dir,
    "resnet18_best.pth"
)

print("모델 저장 위치 :", best_model_path)


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치 :", device)


# ==========================================
# Dataset
# ==========================================

train_loader, val_loader, classes = get_dataloaders(
    args.data_dir,
    args.batch_size
)

print("클래스 수 :", len(classes))
print("Train 이미지 :", len(train_loader.dataset))
print("Validation 이미지 :", len(val_loader.dataset))


# ==========================================
# Model
# ==========================================

model = get_model(
    num_classes=len(classes)
)

model = model.to(device)


# ==========================================
# Loss / Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)


# ==========================================
# Best Model
# ==========================================

best_val_accuracy = 0.0


# ==========================================
# Training
# ==========================================

for epoch in range(args.epochs):

    epoch_start_time = time.time()

    print(f"\nEpoch [{epoch + 1}/{args.epochs}]")

    # --------------------------------------
    # Train
    # --------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_progress = tqdm(
        train_loader,
        desc="Train",
        leave=True
    )

    for images, labels in train_progress:

        images = images.to(device)
        labels = labels.to(device)

        # 이전 gradient 초기화
        optimizer.zero_grad()

        # 순전파
        outputs = model(images)

        # Loss 계산
        loss = criterion(outputs, labels)

        # 역전파
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

        # 현재까지의 정확도
        current_accuracy = (
            100 * train_correct / train_total
        )

        # tqdm 오른쪽에 실시간 정보 표시
        train_progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_accuracy:.2f}%"
        )


    train_accuracy = (
        100 * train_correct / train_total
    )

    average_train_loss = (
        train_loss / len(train_loader)
    )


    # --------------------------------------
    # Validation
    # --------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_progress = tqdm(
        val_loader,
        desc="Validation",
        leave=True
    )

    with torch.no_grad():

        for images, labels in val_progress:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

            current_val_accuracy = (
                100 * val_correct / val_total
            )

            val_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_val_accuracy:.2f}%"
            )


    val_accuracy = (
        100 * val_correct / val_total
    )

    average_val_loss = (
        val_loss / len(val_loader)
    )


    # --------------------------------------
    # Epoch 시간
    # --------------------------------------

    epoch_time = (
        time.time() - epoch_start_time
    )


    # --------------------------------------
    # 결과 출력
    # --------------------------------------

    print(
        f"Train Loss     : "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss       : "
        f"{average_val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Epoch Time     : "
        f"{epoch_time:.2f}초"
    )


    # --------------------------------------
    # Best Model 저장
    # --------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "classes": classes
        },
        best_model_path
    )

    print(
        f"Best Model 저장! "
        f"({val_accuracy:.2f}%)"
    )


print("\n학습 완료")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)