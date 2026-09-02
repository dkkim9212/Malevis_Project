import torch
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

args = parser.parse_args()


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치 :", device)


train_loader, val_loader, classes = get_dataloaders(
    args.data_dir,
    args.batch_size
)


print("클래스 수 :", len(classes))
print("클래스 :", classes)

print("Train 이미지 :", len(train_loader.dataset))
print("Validation 이미지 :", len(val_loader.dataset))


model = get_model(
    num_classes=len(classes)
)

model = model.to(device)

print(model.fc)