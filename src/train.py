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
    unfreeze_layer3_layer4
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
    default=64
)

# FC warm-up
parser.add_argument(
    "--warmup_epochs",
    type=int,
    default=3
)

# Layer3 + Layer4 + FC fine-tuning
parser.add_argument(
    "--finetune_epochs",
    type=int,
    default=25
)

# Warm-up FC LR
parser.add_argument(
    "--head_lr",
    type=float,
    default=0.0003
)

# Layer3 + Layer4 LR
parser.add_argument(
    "--backbone_lr",
    type=float,
    default=0.00001
)

# Fine-tuning FC LR
parser.add_argument(
    "--finetune_head_lr",
    type=float,
    default=0.0003
)

parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0001
)

parser.add_argument(
    "--patience",
    type=int,
    default=7
)

args = parser.parse_args()


# ==========================================
# 3. Save paths
# ==========================================

os.makedirs(
    args.save_dir,
    exist_ok=True
)

best_model_path = os.path.join(
    args.save_dir,
    "resnet18_stage1_best.pth"
)

csv_path = os.path.join(
    args.save_dir,
    "resnet18_stage1_history.csv"
)

print("모델 저장 위치 :", best_model_path)
print("학습 기록 위치 :", csv_path)


# ==========================================
# 4. Device
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("사용 장치 :", device)


# ==========================================
# 5. Dataset
# ==========================================

train_loader, val_loader, classes = get_dataloaders(
    args.data_dir,
    args.batch_size
)

print("클래스 :", classes)
print("클래스 수 :", len(classes))
print("Train 이미지 :", len(train_loader.dataset))
print("Validation 이미지 :", len(val_loader.dataset))

print(
    "Train 원본 분포 :",
    train_loader.dataset.class_counts
)

print(
    "Validation 원본 분포 :",
    val_loader.dataset.class_counts
)

# --------------------------------------------------
# 클래스 순서 검증
#
# 아래 Loss weight는
# 0 = Benign
# 1 = Malware
# 를 기준으로 작성되어 있으므로 반드시 확인
# --------------------------------------------------

print("클래스 순서 :", classes)

assert len(classes) == 2, (
    f"Stage1은 2클래스여야 합니다: {classes}"
)

assert classes[0].lower() == "benign", (
    f"0번 클래스가 benign이 아닙니다: {classes}"
)

assert classes[1].lower() == "malware", (
    f"1번 클래스가 malware가 아닙니다: {classes}"
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
#
# class 0 = Benign
# class 1 = Malware
#
# 이번 실험에서는 기존 weight 6 유지
# ==========================================

class_weights = torch.tensor(
    [6.0, 1.0],
    dtype=torch.float32,
    device=device
)

print("Class Weights :", class_weights)

criterion = nn.CrossEntropyLoss(
    #weight=class_weights,
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
# 9. Metric helper
#
# label 0 = Benign
# label 1 = Malware
# ==========================================

def calculate_metrics(tp, tn, fp, fn):

    accuracy = (
        100.0
        * (tp + tn)
        / max(tp + tn + fp + fn, 1)
    )

    malware_precision = (
        100.0
        * tp
        / max(tp + fp, 1)
    )

    malware_recall = (
        100.0
        * tp
        / max(tp + fn, 1)
    )

    malware_f1 = (
        2
        * malware_precision
        * malware_recall
        / max(
            malware_precision + malware_recall,
            1e-12
        )
    )

    benign_recall = (
        100.0
        * tn
        / max(tn + fp, 1)
    )

    # --------------------------------------------------
    # 두 클래스 Recall의 평균
    #
    # 데이터 불균형이 있는 Stage1에서는
    # 일반 Accuracy보다 모델 균형을 확인하기 좋음
    # --------------------------------------------------

    balanced_accuracy = (
        malware_recall + benign_recall
    ) / 2.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "malware_precision": malware_precision,
        "malware_recall": malware_recall,
        "malware_f1": malware_f1,
        "benign_recall": benign_recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# ==========================================
# 10. Train
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

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        current_accuracy = (
            100.0
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
        100.0
        * correct
        / total
    )

    return average_loss, accuracy


# ==========================================
# 11. Validation
# ==========================================

def validate(
    model,
    val_loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

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

            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            # ------------------------------------------
            # TP
            # 실제 Malware / 예측 Malware
            # ------------------------------------------

            tp += (
                (predicted == 1)
                & (labels == 1)
            ).sum().item()

            # ------------------------------------------
            # TN
            # 실제 Benign / 예측 Benign
            # ------------------------------------------

            tn += (
                (predicted == 0)
                & (labels == 0)
            ).sum().item()

            # ------------------------------------------
            # FP
            # 실제 Benign / 예측 Malware
            # ------------------------------------------

            fp += (
                (predicted == 1)
                & (labels == 0)
            ).sum().item()

            # ------------------------------------------
            # FN
            # 실제 Malware / 예측 Benign
            # ------------------------------------------

            fn += (
                (predicted == 0)
                & (labels == 1)
            ).sum().item()

            metrics = calculate_metrics(
                tp,
                tn,
                fp,
                fn
            )

            progress.set_postfix(
                acc=f"{metrics['accuracy']:.2f}%",
                bal=f"{metrics['balanced_accuracy']:.2f}%",
                recall=f"{metrics['malware_recall']:.2f}%"
            )

    average_loss = (
        running_loss
        / len(val_loader)
    )

    metrics = calculate_metrics(
        tp,
        tn,
        fp,
        fn
    )

    return average_loss, metrics


# ==========================================
# 12. CSV
# ==========================================

history = []

# --------------------------------------------------
# 전체 학습 과정에서 가장 좋은 Balanced Accuracy
# --------------------------------------------------

best_val_score = -1.0

global_epoch = 0


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
                "balanced_accuracy",

                "malware_precision",
                "malware_recall",
                "malware_f1",

                "benign_recall",

                "tp",
                "tn",
                "fp",
                "fn",

                "backbone_lr",
                "head_lr",

                "epoch_time"
            ]
        )

        writer.writeheader()

        writer.writerows(history)


# ==========================================
# 13. Best Model 저장
# ==========================================

def save_best_model(
    epoch,
    stage,
    val_metrics,
    optimizer
):

    torch.save(
        {
            "epoch": epoch,

            "stage": stage,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_accuracy":
                val_metrics["accuracy"],

            "balanced_accuracy":
                val_metrics["balanced_accuracy"],

            "malware_precision":
                val_metrics["malware_precision"],

            "malware_recall":
                val_metrics["malware_recall"],

            "malware_f1":
                val_metrics["malware_f1"],

            "benign_recall":
                val_metrics["benign_recall"],

            "classes":
                classes
        },

        best_model_path
    )


# ==========================================
# 14. Validation 결과 출력
# ==========================================

def print_val_metrics(
    val_loss,
    metrics
):

    print(
        f"Val Loss          : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy      : "
        f"{metrics['accuracy']:.2f}%"
    )

    print(
        f"Balanced Accuracy : "
        f"{metrics['balanced_accuracy']:.2f}%"
    )

    print(
        f"Malware Precision : "
        f"{metrics['malware_precision']:.2f}%"
    )

    print(
        f"Malware Recall    : "
        f"{metrics['malware_recall']:.2f}%"
    )

    print(
        f"Malware F1        : "
        f"{metrics['malware_f1']:.2f}%"
    )

    print(
        f"Benign Recall     : "
        f"{metrics['benign_recall']:.2f}%"
    )

    print(
        "Confusion Matrix  : "
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )


# ==================================================
# Phase 1
# FC Warm-up
# ==================================================

print("\n" + "=" * 60)

print(
    "Phase 1 : FC Layer Warm-up"
)

print("=" * 60)


# ==========================================
# Backbone Freeze
# ==========================================

freeze_backbone(model)


# ==========================================
# FC Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=args.head_lr,
    weight_decay=args.weight_decay
)


# ==========================================
# Warm-up
# ==========================================

for epoch in range(
    args.warmup_epochs
):

    global_epoch += 1

    start_time = time.time()

    print(
        f"\nWarm-up "
        f"[{epoch + 1}/{args.warmup_epochs}]"
    )


    # ------------------------------------------
    # Train
    # ------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )


    # ------------------------------------------
    # Validation
    # ------------------------------------------

    val_loss, val_metrics = validate(
        model,
        val_loader,
        criterion,
        device
    )


    epoch_time = (
        time.time()
        - start_time
    )


    # ------------------------------------------
    # 결과 출력
    # ------------------------------------------

    print(
        f"Train Loss        : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy    : "
        f"{train_accuracy:.2f}%"
    )

    print_val_metrics(
        val_loss,
        val_metrics
    )

    print(
        f"Epoch Time        : "
        f"{epoch_time:.2f}초"
    )


    # ------------------------------------------
    # History
    # ------------------------------------------

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
            val_metrics["accuracy"],

        "balanced_accuracy":
            val_metrics["balanced_accuracy"],

        "malware_precision":
            val_metrics["malware_precision"],

        "malware_recall":
            val_metrics["malware_recall"],

        "malware_f1":
            val_metrics["malware_f1"],

        "benign_recall":
            val_metrics["benign_recall"],

        "tp":
            val_metrics["tp"],

        "tn":
            val_metrics["tn"],

        "fp":
            val_metrics["fp"],

        "fn":
            val_metrics["fn"],

        "backbone_lr":
            0,

        "head_lr":
            args.head_lr,

        "epoch_time":
            epoch_time
    })


    save_history()


    # ==========================================
    # Best Model
    #
    # 기존 Malware F1 기준에서
    # Balanced Accuracy 기준으로 변경
    # ==========================================

    if (
        val_metrics["balanced_accuracy"]
        > best_val_score
    ):

        best_val_score = (
            val_metrics["balanced_accuracy"]
        )

        save_best_model(
            global_epoch,
            "FC_WARMUP",
            val_metrics,
            optimizer
        )

        print(
            f"Best Model 저장! "
            f"(Balanced Accuracy="
            f"{best_val_score:.2f}%)"
        )


# ==================================================
# Phase 2
# Layer3 + Layer4 + FC Fine-tuning
# ==================================================

print("\n" + "=" * 60)

print(
    "Phase 2 : Layer3 + Layer4 + FC Fine-tuning"
)

print("=" * 60)


# ==========================================
# Layer3 + Layer4 Unfreeze
# ==========================================

unfreeze_layer3_layer4(model)


# ==========================================
# Fine-tuning Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    [
        {
            "params":
                list(model.layer3.parameters())
                + list(model.layer4.parameters()),

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

    weight_decay=
        args.weight_decay
)


# ==========================================
# LR Scheduler
# ==========================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
)


# ==========================================
# Fine-tuning Early Stopping
#
# Warm-up 최고 기록과 별개로
# Fine-tuning 자체의 개선 여부를 확인
# ==========================================

early_stop_count = 0

best_finetune_score = -1.0


# ==========================================
# Fine-tuning
# ==========================================

for epoch in range(
    args.finetune_epochs
):

    global_epoch += 1

    start_time = time.time()

    print(
        f"\nFine-tuning "
        f"[{epoch + 1}/{args.finetune_epochs}]"
    )


    # ------------------------------------------
    # Train
    # ------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )


    # ------------------------------------------
    # Validation
    # ------------------------------------------

    val_loss, val_metrics = validate(
        model,
        val_loader,
        criterion,
        device
    )


    # ------------------------------------------
    # Scheduler
    # ------------------------------------------

    scheduler.step(
        val_loss
    )


    backbone_lr = (
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


    # ------------------------------------------
    # 결과 출력
    # ------------------------------------------

    print(
        f"Train Loss        : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy    : "
        f"{train_accuracy:.2f}%"
    )

    print_val_metrics(
        val_loss,
        val_metrics
    )

    print(
        f"Backbone LR       : "
        f"{backbone_lr:.8f}"
    )

    print(
        f"Head LR           : "
        f"{head_lr:.8f}"
    )

    print(
        f"Epoch Time        : "
        f"{epoch_time:.2f}초"
    )


    # ------------------------------------------
    # History
    # ------------------------------------------

    history.append({

        "epoch":
            global_epoch,

        "stage":
            "LAYER3_4_FINETUNE",

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_metrics["accuracy"],

        "balanced_accuracy":
            val_metrics["balanced_accuracy"],

        "malware_precision":
            val_metrics["malware_precision"],

        "malware_recall":
            val_metrics["malware_recall"],

        "malware_f1":
            val_metrics["malware_f1"],

        "benign_recall":
            val_metrics["benign_recall"],

        "tp":
            val_metrics["tp"],

        "tn":
            val_metrics["tn"],

        "fp":
            val_metrics["fp"],

        "fn":
            val_metrics["fn"],

        "backbone_lr":
            backbone_lr,

        "head_lr":
            head_lr,

        "epoch_time":
            epoch_time
    })


    save_history()


    # ==========================================
    # Fine-tuning Early Stopping
    # ==========================================

    current_score = (
        val_metrics["balanced_accuracy"]
    )


    # ------------------------------------------
    # Fine-tuning 자체 최고 기록 갱신
    # ------------------------------------------

    if (
        current_score
        > best_finetune_score
    ):

        best_finetune_score = (
            current_score
        )

        early_stop_count = 0

        print(
            f"Fine-tuning "
            f"Balanced Accuracy 개선! "
            f"({best_finetune_score:.2f}%)"
        )


        # --------------------------------------
        # 전체 Warm-up + Fine-tuning 중
        # 가장 좋은 모델일 때만 저장
        # --------------------------------------

        if (
            current_score
            > best_val_score
        ):

            best_val_score = (
                current_score
            )

            save_best_model(
                global_epoch,
                "LAYER3_4_FINETUNE",
                val_metrics,
                optimizer
            )

            print(
                f"Best Model 저장! "
                f"(Balanced Accuracy="
                f"{best_val_score:.2f}%)"
            )


    # ------------------------------------------
    # 개선 없음
    # ------------------------------------------

    else:

        early_stop_count += 1

        print(
            f"Balanced Accuracy 개선 없음 "
            f"({early_stop_count}/"
            f"{args.patience})"
        )


    # ------------------------------------------
    # Early Stopping
    # ------------------------------------------

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

print(
    "\n" + "=" * 60
)

print(
    "학습 완료"
)

print(
    "=" * 60
)

print(
    f"Best Balanced Accuracy : "
    f"{best_val_score:.2f}%"
)

print(
    f"Best Model : "
    f"{best_model_path}"
)

print(
    f"History CSV : "
    f"{csv_path}"
)