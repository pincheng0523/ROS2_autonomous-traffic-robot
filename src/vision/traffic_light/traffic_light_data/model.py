import os
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(data_root: str, batch_size: int = 32):
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"
    test_dir = Path(data_root) / "test"

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(num_classes: int):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return running_loss / total, running_correct / total, all_labels, all_preds


def main():
    data_root = "/home/cpc/turtlebot_project_ws/src/vision/dataset/traffic_light/traffic_light_data"   
    save_dir = Path("./runs/traffic_light_efficientnet")
    save_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 15
    batch_size = 32
    lr = 1e-4

    device = get_device()
    print("Device:", device)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        data_root, batch_size=batch_size
    )

    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_dir / "best.pt")

    model.load_state_dict(best_weights)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": class_names,
        },
        save_dir / "final_with_classes.pt"
    )


if __name__ == "__main__":
    main()
