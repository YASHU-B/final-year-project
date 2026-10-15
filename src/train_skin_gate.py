"""Train lightweight Neural Skin-Presence Gate to reject non-skin Out-Of-Distribution images."""

import os
import ssl
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "skin_gate"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT_PATH = CHECKPOINTS_DIR / "skin_gate_classifier.pt"

ssl._create_default_https_context = ssl._create_unverified_context


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SkinPresenceClassifier(nn.Module):
    """Binary classifier: Class 0 = Non-Skin / OOD, Class 1 = Human Skin / Lesion."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(1280, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.avgpool(feat).flatten(1)
        return self.classifier(pooled)


def train():
    device = resolve_device()
    print(f"Training Skin-Presence Gate on device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(DATA_DIR / "val"), transform=val_transform)

    print(f"Classes: {train_dataset.class_to_idx}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Backbone
    base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Freeze feature backbone
    for param in base_model.features.parameters():
        param.requires_grad = False

    # Unfreeze top conv block for adaptation
    for param in base_model.features[7].parameters():
        param.requires_grad = True

    model = SkinPresenceClassifier(base_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_val_acc = 0.0
    epochs = 6

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_acc = v_correct / max(v_total, 1)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {running_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "classes": ["non_skin", "skin"],
                "best_val_acc": best_val_acc,
            }, MODEL_OUT_PATH)

    print(f"\nTrained Skin Gate saved to: {MODEL_OUT_PATH} (Best Val Acc: {best_val_acc*100:.1f}%)")


if __name__ == "__main__":
    train()
