import os
import csv
import time
import random
import argparse

import torch
import torch.nn as nn

from tqdm.auto import tqdm

from dataset import get_dataloaders

from model import (
    get_model,
    freeze_backbone,
    unfreeze_layer4
)


# ==========================================
# 1. Random Seed
# ==========================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ==========================================
# 2. Arguments
# ==========================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--data_dir",
    type=str,
    required=True
)


parser.add_argument(
    "--save_dir",
    type=str,
    default="./models"
)


parser.add_argument(
    "--batch_size",
    type=int,
    default=32
)


# Stage 1 : FC만 학습
parser.add_argument(
    "--warmup_epochs",
    type=int,
    default=2
)


# Stage 2 : Layer4 + FC 학습
parser.add_argument(
    "--finetune_epochs",
    type=int,
    default=10
)


# Stage 1 FC Learning Rate
parser.add_argument(
    "--head_lr",
    type=float,
    default=0.0003
)


# Stage 2 Layer4 Learning Rate
parser.add_argument(
    "--backbone_lr",
    type=float,
    default=0.00001
)


# Stage 2 FC Learning Rate
parser.add_argument(
    "--finetune_head_lr",
    type=float,
    default=0.0001
)


parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0001
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
    "resnet18_exp3_best.pth"
)


csv_path = os.path.join(
    args.save_dir,
    "resnet18_exp3_history.csv"
)


print(
    "모델 저장 위치 :",
    best_model_path
)

print(
    "학습 기록 위치 :",
    csv_path
)


# ==========================================
# 4. Device
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "사용 장치 :",
    device
)


# ==========================================
# 5. Dataset
# ==========================================

train_loader, val_loader, classes = (
    get_dataloaders(
        args.data_dir,
        args.batch_size
    )
)


print(
    "클래스 수 :",
    len(classes)
)

print(
    "Train 이미지 :",
    len(train_loader.dataset)
)

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
# 8. AMP
# ==========================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)


# ==========================================
# 9. 기록용 변수
# ==========================================

history = []

best_val_accuracy = 0.0

global_epoch = 0


# ==========================================
# 공통 Train 함수
# ==========================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    progress = tqdm(
        train_loader,
        desc="Train",
        leave=True
    )


    for images, labels in progress:

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


        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        running_loss += (
            loss.item()
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


        current_accuracy = (
            100
            * correct
            / total
        )


        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_accuracy:.2f}%"
        )


    average_loss = (
        running_loss
        / len(train_loader)
    )


    accuracy = (
        100
        * correct
        / total
    )


    return (
        average_loss,
        accuracy
    )


# ==========================================
# 공통 Validation 함수
# ==========================================

def validate(
    model,
    val_loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0


    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=True
    )


    with torch.no_grad():

        for images, labels in progress:

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


            running_loss += (
                loss.item()
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


            current_accuracy = (
                100
                * correct
                / total
            )


            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_accuracy:.2f}%"
            )


    average_loss = (
        running_loss
        / len(val_loader)
    )


    accuracy = (
        100
        * correct
        / total
    )


    return (
        average_loss,
        accuracy
    )


# ==========================================
# CSV 저장 함수
# ==========================================

def save_history():

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
                "stage",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "layer4_lr",
                "head_lr",
                "epoch_time"
            ]
        )


        writer.writeheader()

        writer.writerows(
            history
        )


# ==========================================
# Best Model 저장 함수
# ==========================================

def save_best_model(
    epoch,
    stage,
    val_accuracy,
    optimizer
):

    torch.save(
        {
            "epoch":
                epoch,

            "stage":
                stage,

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


# ==================================================
#
# Stage 1
# FC만 학습
#
# ==================================================

print("\n")
print("=" * 50)
print("Stage 1 : FC Layer Warm-up")
print("=" * 50)


freeze_backbone(
    model
)


# FC Layer만 optimizer에 전달
optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=args.head_lr,
    weight_decay=args.weight_decay
)


for epoch in range(
    args.warmup_epochs
):

    global_epoch += 1

    start_time = time.time()


    print(
        f"\nWarm-up "
        f"[{epoch + 1}/"
        f"{args.warmup_epochs}]"
    )


    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
    )


    val_loss, val_accuracy = (
        validate(
            model,
            val_loader,
            criterion,
            device
        )
    )


    epoch_time = (
        time.time()
        - start_time
    )


    print(
        f"Train Loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss       : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Epoch Time     : "
        f"{epoch_time:.2f}초"
    )


    history.append({
        "epoch":
            global_epoch,

        "stage":
            "FC_WARMUP",

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "layer4_lr":
            0,

        "head_lr":
            args.head_lr,

        "epoch_time":
            epoch_time
    })


    save_history()


    if (
        val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )


        save_best_model(
            global_epoch,
            "FC_WARMUP",
            val_accuracy,
            optimizer
        )


        print(
            f"Best Model 저장! "
            f"({val_accuracy:.2f}%)"
        )


# ==================================================
#
# Stage 2
# Layer4 + FC Fine-tuning
#
# ==================================================

print("\n")
print("=" * 50)
print("Stage 2 : Layer4 + FC Fine-tuning")
print("=" * 50)


unfreeze_layer4(
    model
)


# ==========================================
# 서로 다른 Learning Rate 적용
# ==========================================

optimizer = torch.optim.AdamW(

    [
        {
            "params":
                model.layer4.parameters(),

            "lr":
                args.backbone_lr
        },

        {
            "params":
                model.fc.parameters(),

            "lr":
                args.finetune_head_lr
        }
    ],

    weight_decay=args.weight_decay
)


# ==========================================
# Scheduler
# ==========================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )
)


early_stop_count = 0


for epoch in range(
    args.finetune_epochs
):

    global_epoch += 1

    start_time = time.time()


    print(
        f"\nFine-tuning "
        f"[{epoch + 1}/"
        f"{args.finetune_epochs}]"
    )


    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
    )


    val_loss, val_accuracy = (
        validate(
            model,
            val_loader,
            criterion,
            device
        )
    )


    # ======================================
    # Scheduler
    # ======================================

    scheduler.step(
        val_accuracy
    )


    layer4_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    head_lr = (
        optimizer
        .param_groups[1]["lr"]
    )


    epoch_time = (
        time.time()
        - start_time
    )


    # ======================================
    # 결과 출력
    # ======================================

    print(
        f"Train Loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss       : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Layer4 LR      : "
        f"{layer4_lr:.8f}"
    )

    print(
        f"Head LR        : "
        f"{head_lr:.8f}"
    )

    print(
        f"Epoch Time     : "
        f"{epoch_time:.2f}초"
    )


    # ======================================
    # 기록
    # ======================================

    history.append({
        "epoch":
            global_epoch,

        "stage":
            "LAYER4_FINETUNE",

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "layer4_lr":
            layer4_lr,

        "head_lr":
            head_lr,

        "epoch_time":
            epoch_time
    })


    save_history()


    # ======================================
    # Best Model
    # ======================================

    if (
        val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )

        early_stop_count = 0


        save_best_model(
            global_epoch,
            "LAYER4_FINETUNE",
            val_accuracy,
            optimizer
        )


        print(
            f"Best Model 저장! "
            f"({val_accuracy:.2f}%)"
        )


    else:

        early_stop_count += 1


        print(
            f"성능 개선 없음 "
            f"({early_stop_count}/"
            f"{args.patience})"
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
# 종료
# ==========================================

print("\n")
print("=" * 50)
print("학습 완료")
print("=" * 50)


print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Model : "
    f"{best_model_path}"
)

print(
    f"History CSV : "
    f"{csv_path}"
)