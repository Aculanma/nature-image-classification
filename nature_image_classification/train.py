from pathlib import Path
from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import os


# -------------------
# Dataset
# -------------------
class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# -------------------
# Training function
# -------------------
def train_model(
    train_dir,
    val_dir,
    test_dir,
    epochs,
    batch_size,
    lr,
    seed,
    output_dir,
    model_type="baseline",
):
    print("Training started")
    print(f"Model type: {model_type}")
    print(f"Random seed: {seed}")

    # -------------------
    # Load data
    # -------------------
    data_dir = Path(train_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    image_paths = []
    labels = []
    for i, cls in enumerate(class_names):
        cls_files = list((data_dir / cls).glob("*.jpg"))
        image_paths.extend(cls_files)
        labels.extend([i] * len(cls_files))

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # -------------------
    # Train/Val split
    # -------------------
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.15, stratify=labels, random_state=seed
    )

    # -------------------
    # Load test data
    # -------------------
    test_dir = Path(test_dir)
    X_test = []
    y_test = []

    for i, cls in enumerate(class_names):
        cls_files = list((test_dir / cls).glob("*.jpg"))
        X_test.extend(cls_files)
        y_test.extend([i] * len(cls_files))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # -------------------
    # Baseline model
    # -------------------
    if model_type == "baseline":
        print("Training baseline model (LogisticRegression)")
        X_train_flat = np.array(
            [np.array(Image.open(p).resize((150, 150))).flatten() for p in X_train]
        )
        X_val_flat = np.array(
            [np.array(Image.open(p).resize((150, 150))).flatten() for p in X_val]
        )

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_val_flat)
        acc = accuracy_score(y_val, y_pred)
        print(f"Baseline accuracy: {acc:.4f}")

        # ---- Test accuracy ----
        X_test_flat = np.array(
            [np.array(Image.open(p).resize((150, 150))).flatten() for p in X_test]
        )

        y_test_pred = clf.predict(X_test_flat)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Baseline TEST accuracy: {test_acc:.4f}")

        os.makedirs(output_dir, exist_ok=True)

        metrics = {
            "model": "baseline",
            "val_accuracy": acc,
            "test_accuracy": test_acc,
        }

        with open(Path(output_dir) / "metrics_baseline.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print("Baseline metrics saved")
    # -------------------
    # ResNet50 model
    # -------------------
    elif model_type == "resnet50":
        print("Training ResNet50 model")
        transform = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
            ]
        )
        train_dataset = SimpleImageDataset(X_train, y_train, transform=transform)
        val_dataset = SimpleImageDataset(X_val, y_val, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = models.wide_resnet50_2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr)

        for epoch in range(epochs):
            # --- Training ---
            model.train()
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                output = model(imgs)
                loss = criterion(output, lbls)
                loss.backward()
                optimizer.step()

            # --- Validation accuracy ---
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    output = model(imgs)
                    preds = output.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(lbls.cpu().numpy())
            acc = accuracy_score(val_labels, val_preds)
            print(f"Epoch {epoch + 1}/{epochs} done - Validation Accuracy: {acc:.4f}")
        # -------------------
        # Test evaluation
        # -------------------
        model.eval()
        correct = 0
        total = 0

        test_dataset = SimpleImageDataset(X_test, y_test, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

        test_acc = correct / total
        print(f"ResNet50 TEST accuracy: {test_acc:.4f}")
        print("ResNet50 training finished")

        os.makedirs(output_dir, exist_ok=True)

        metrics = {
            "model": "resnet50",
            "val_accuracy": acc,
            "test_accuracy": test_acc,
        }

        with open(Path(output_dir) / "metrics_resnet50.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print("ResNet50 metrics saved")
        model_path = Path(output_dir) / "resnet50_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


# -------------------
# Hydra entrypoint
# -------------------
@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="baseline",
)
def main(cfg: DictConfig):
    train_model(
        train_dir=cfg.data.train_dir,
        val_dir=cfg.data.val_dir,
        test_dir=cfg.data.test_dir,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        seed=cfg.training.seed,
        output_dir=cfg.output.dir,
        model_type=cfg.model.type,
    )


if __name__ == "__main__":
    main()
