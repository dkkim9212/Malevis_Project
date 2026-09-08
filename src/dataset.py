import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms


class MaleVisBinaryDataset(Dataset):
    """
    MaleVis 26개 원본 폴더를 그대로 사용하면서 Stage1용 2-class로 재매핑.

    Other -> 0 (Benign)
    나머지 25개 클래스 -> 1 (Malware)
    """

    def __init__(self, root, transform=None):
        self.base_dataset = datasets.ImageFolder(
            root=root,
            transform=transform
        )

        self.classes = ["Benign", "Malware"]
        self.class_to_idx = {
            "Benign": 0,
            "Malware": 1
        }

        idx_to_class = {
            idx: class_name
            for class_name, idx
            in self.base_dataset.class_to_idx.items()
        }

        self.targets = []

        for _, original_label in self.base_dataset.samples:
            class_name = idx_to_class[original_label]

            if class_name.lower() == "other":
                binary_label = 0
            else:
                binary_label = 1

            self.targets.append(binary_label)

        self.class_counts = {
            "Benign": self.targets.count(0),
            "Malware": self.targets.count(1)
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        label = self.targets[index]

        return image, label


def make_balanced_sampler(targets, balance_power=0.5):
    """
    25:1 수준의 불균형을 완화한다.

    완전한 50:50 오버샘플링은 Benign 표본을 지나치게 반복할 수 있으므로
    기본값은 sqrt inverse-frequency(1/sqrt(count))를 사용한다.
    """

    labels = torch.tensor(targets, dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=2).float()

    class_weights = 1.0 / torch.pow(
        class_counts.clamp_min(1.0),
        balance_power
    )

    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True
    )


def get_dataloaders(data_dir, batch_size=64):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = MaleVisBinaryDataset(
        root=f"{data_dir}/train",
        transform=train_transform
    )

    val_dataset = MaleVisBinaryDataset(
        root=f"{data_dir}/val",
        transform=val_transform
    )

    train_sampler = make_balanced_sampler(
        train_dataset.targets,
        balance_power=1.0
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

    return train_loader, val_loader, train_dataset.classes
