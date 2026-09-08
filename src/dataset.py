import os

import torch

from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision import (
    datasets,
    transforms
)


# ==========================================
# Binary Dataset
# ==========================================

class MaleVisBinaryDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples

        self.transform = transform

        self.classes = [
            "Benign",
            "Malware"
        ]

        self.targets = [
            label
            for _, label
            in samples
        ]

        self.class_counts = {
            "Benign":
                self.targets.count(0),

            "Malware":
                self.targets.count(1)
        }


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        image_path, label = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        return (
            image,
            label
        )


# ==========================================
# 기존 train + val 전체 수집
# ==========================================

def collect_all_samples(
    data_dir
):

    all_samples = []

    for split_name in [
        "train",
        "val"
    ]:

        split_dir = os.path.join(
            data_dir,
            split_name
        )

        dataset = datasets.ImageFolder(
            root=split_dir
        )

        idx_to_class = {
            idx: class_name
            for class_name, idx
            in dataset.class_to_idx.items()
        }

        for (
            image_path,
            original_label
        ) in dataset.samples:

            class_name = (
                idx_to_class[
                    original_label
                ]
            )

            # Other = 정상
            if (
                class_name.lower()
                == "other"
            ):

                binary_label = 0

            # 나머지 25개 = 악성
            else:

                binary_label = 1

            all_samples.append(
                (
                    image_path,
                    binary_label
                )
            )

    return all_samples


# ==========================================
# 70 / 15 / 15 Stratified Split
# ==========================================

def split_samples(
    samples,
    seed=42
):

    labels = [
        label
        for _, label
        in samples
    ]

    # --------------------------------------
    # 70% Train
    # 30% Temporary
    # --------------------------------------

    train_samples, temp_samples = (
        train_test_split(
            samples,

            test_size=0.30,

            random_state=seed,

            stratify=labels
        )
    )

    temp_labels = [
        label
        for _, label
        in temp_samples
    ]

    # --------------------------------------
    # 나머지 30%를 절반씩
    # Val 15%
    # Test 15%
    # --------------------------------------

    val_samples, test_samples = (
        train_test_split(
            temp_samples,

            test_size=0.50,

            random_state=seed,

            stratify=temp_labels
        )
    )

    return (
        train_samples,
        val_samples,
        test_samples
    )


# ==========================================
# Weighted Sampler
# ==========================================

def make_balanced_sampler(
    targets,
    balance_power=0.7
):

    labels = torch.tensor(
        targets,
        dtype=torch.long
    )

    class_counts = torch.bincount(
        labels,
        minlength=2
    ).float()

    class_weights = (
        1.0
        /
        torch.pow(
            class_counts.clamp_min(
                1.0
            ),
            balance_power
        )
    )

    sample_weights = (
        class_weights[
            labels
        ]
    )

    sampler = (
        WeightedRandomSampler(
            weights=
                sample_weights.double(),

            num_samples=
                len(sample_weights),

            replacement=True
        )
    )

    return sampler


# ==========================================
# DataLoader
# ==========================================

def get_dataloaders(
    data_dir,
    batch_size=64
):

    transform = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


    # ======================================
    # 전체 데이터 수집
    # ======================================

    all_samples = (
        collect_all_samples(
            data_dir
        )
    )


    # ======================================
    # Train / Val / Test
    # ======================================

    (
        train_samples,
        val_samples,
        test_samples
    ) = split_samples(
        all_samples,
        seed=42
    )


    train_dataset = (
        MaleVisBinaryDataset(
            train_samples,
            transform=transform
        )
    )


    val_dataset = (
        MaleVisBinaryDataset(
            val_samples,
            transform=transform
        )
    )


    test_dataset = (
        MaleVisBinaryDataset(
            test_samples,
            transform=transform
        )
    )


    # ======================================
    # Train sampler
    # ======================================

    train_sampler = (
        make_balanced_sampler(
            train_dataset.targets,
            balance_power=0.7
        )
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        sampler=train_sampler,

        num_workers=2,

        pin_memory=True,

        persistent_workers=True
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=2,

        pin_memory=True,

        persistent_workers=True
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=2,

        pin_memory=True,

        persistent_workers=True
    )


    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )