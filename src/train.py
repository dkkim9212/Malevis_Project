import os
import csv
import time
import random
import argparse

import torch
import torch.nn as nn

from tqdm.auto import tqdm

from dataset import get_dataloaders
from model import get_model


# ==========================================
# 1. Random Seed
# ==========================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ==========================================
# 2. 실행 옵션
# ==========================================

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
    default=12
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.00003
)

parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0001
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="./models"
)

parser.add_argument(
    "--patience",
    type=int,
    default=4
)

args = parser.parse_args()


# ==========================================
# 3. 저장 경로
# ==========================================

os.makedirs(
    args.save_dir,
    exist_ok=True
)

best_model_path = os.path.join(
    args.save_dir,
    "resnet18_best.pth"
)

csv_path = os.path.join(
    args.save_dir,
    "resnet18_history.csv"
)

print("모델 저장 위치 :", best_model_path)
print("학습 기록 위치 :", csv_path)


# ==========================================
# 4. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("사용 장치 :", device)


# ==========================================
# 5. Dataset
# ==========================================

train_loader, val_loader, classes = (
    get_dataloaders(
        args.data_dir,
        args.batch_size
    )
)

print("클래스 수 :", len(classes))
print("Train 이미지 :", len(train_loader.dataset))
print(
    "Validation 이미지 :",
    len(val_loader.dataset)
)


# ==========================================
# 6. Model
# ==========================================

model = get_model(
    num_classes=len(classes)
)

model = model.to(device)


# ==========================================
# 7. Loss
# ==========================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ==========================================
# 8. Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)


# ==========================================
# 9. Learning Rate Scheduler
# ==========================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )
)


# ==========================================
# 10. AMP
# ==========================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)


# ==========================================
# 11. 학습 기록
# ==========================================

best_val_accuracy = 0.0

early_stop_count = 0

history = []


# ==========================================
# 12. Training
# ==========================================

for epoch in range(args.epochs):

    epoch_start_time = time.time()

    print(
        f"\nEpoch "
        f"[{epoch + 1}/{args.epochs}]"
    )


    # ======================================
    # Train
    # ======================================

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

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ==============================
        # Mixed Precision Forward
        # ==============================

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        # ==============================
        # Backpropagation
        # ==============================

        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )

        scaler.update()


        # ==============================
        # 결과 계산
        # ==============================

        train_loss += (
            loss.item()
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        train_total += (
            labels.size(0)
        )


        train_correct += (
            predicted == labels
        ).sum().item()


        current_accuracy = (
            100
            * train_correct
            / train_total
        )


        train_progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_accuracy:.2f}%"
        )


    average_train_loss = (
        train_loss
        / len(train_loader)
    )


    train_accuracy = (
        100
        * train_correct
        / train_total
    )


    # ======================================
    # Validation
    # ======================================

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

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            val_loss += (
                loss.item()
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += (
                labels.size(0)
            )


            val_correct += (
                predicted == labels
            ).sum().item()


            current_val_accuracy = (
                100
                * val_correct
                / val_total
            )


            val_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_val_accuracy:.2f}%"
            )


    average_val_loss = (
        val_loss
        / len(val_loader)
    )


    val_accuracy = (
        100
        * val_correct
        / val_total
    )


    # ======================================
    # Scheduler
    # ======================================

    scheduler.step(
        val_accuracy
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ======================================
    # Epoch 시간
    # ======================================

    epoch_time = (
        time.time()
        - epoch_start_time
    )


    # ======================================
    # 결과 출력
    # ======================================

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
        f"Learning Rate  : "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time     : "
        f"{epoch_time:.2f}초"
    )


    # ======================================
    # CSV 기록
    # ======================================

    history.append({
        "epoch":
            epoch + 1,

        "train_loss":
            average_train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            average_val_loss,

        "val_accuracy":
            val_accuracy,

        "learning_rate":
            current_lr,

        "epoch_time":
            epoch_time
    })


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
                "epoch_time"
            ]
        )

        writer.writeheader()

        writer.writerows(
            history
        )


    # ======================================
    # Best Model 저장
    # ======================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        early_stop_count = 0


        torch.save(
            {
                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "classes":
                    classes
            },
            best_model_path
        )


        print(
            f"Best Model 저장! "
            f"({val_accuracy:.2f}%)"
        )


    else:

        early_stop_count += 1

        print(
            f"성능 개선 없음 "
            f"({early_stop_count}"
            f"/{args.patience})"
        )


    # ======================================
    # Early Stopping
    # ======================================

    if (
        early_stop_count
        >= args.patience
    ):

        print(
            "\nEarly Stopping!"
        )

        break


# ==========================================
# 13. 학습 종료
# ==========================================

print("\n학습 완료")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Model: "
    f"{best_model_path}"
)

print(
    f"History CSV: "
    f"{csv_path}"
)