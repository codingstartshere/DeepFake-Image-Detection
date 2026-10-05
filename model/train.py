"""
Training pipeline for the CIFAKE ConvNeXt-Tiny deepfake detector.

Project location:
    model/train.py

Expected dataset structure:
    data/
    ├── train/
    │   ├── REAL/
    │   └── FAKE/
    └── val/
        ├── REAL/
        └── FAKE/

The dataset loading logic remains in model/dataset.py.
The model architecture remains in model/convnext.py.

This file is responsible for:
    - Creating train/validation DataLoaders
    - Creating the ConvNeXt-Tiny model
    - Training the model
    - Calculating training/validation loss and accuracy
    - Saving the best checkpoint to model/weights/
    - Saving the latest checkpoint to model/weights/

Class mapping:
    0 -> REAL
    1 -> FAKE
"""

from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

WEIGHTS_DIR = PROJECT_ROOT / "model" / "weights"
BEST_MODEL_PATH = WEIGHTS_DIR / "convnext_tiny_best.pth"
LAST_MODEL_PATH = WEIGHTS_DIR / "convnext_tiny_last.pth"


# ---------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------

NUM_CLASSES = 2
IMAGE_SIZE = 32

BATCH_SIZE = 64
NUM_EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device() -> torch.device:
    """Select the best available PyTorch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Dataset imports
# ---------------------------------------------------------------------

# Allow this file to be executed directly with:
#     python model/train.py
#
# while still allowing:
#     from model.train import train
#
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import CIFAKEDataset  # noqa: E402
from model.convnext import ConvNeXtTiny  # noqa: E402


# ---------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------

def create_dataloaders():
    """
    Create training and validation DataLoaders.

    CIFAKEDataset is responsible for image loading, transforms,
    and label generation.
    """

    train_dataset = CIFAKEDataset(
        root_dir=TRAIN_DIR,
        image_size=IMAGE_SIZE,
        train=True,
    )

    val_dataset = CIFAKEDataset(
        root_dir=VAL_DIR,
        image_size=IMAGE_SIZE,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, val_dataset, train_loader, val_loader


# ---------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Train the model for one complete epoch."""

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ---------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evaluate the model on the validation set."""

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)

        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
):
    """Save model and training state."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "class_to_idx": {
            "REAL": 0,
            "FAKE": 1,
        },
    }

    torch.save(checkpoint, path)


# ---------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------

def train():
    """Run the complete ConvNeXt-Tiny training pipeline."""

    set_seed()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 60)
    print("CIFAKE ConvNeXt-Tiny Training")
    print("=" * 60)
    print(f"Device:       {device}")
    print(f"Train path:   {TRAIN_DIR}")
    print(f"Val path:     {VAL_DIR}")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Epochs:       {NUM_EPOCHS}")
    print(f"Learning rate:{LEARNING_RATE}")
    print(f"Image size:   {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("=" * 60)

    train_dataset, val_dataset, train_loader, val_loader = (
        create_dataloaders()
    )

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = ConvNeXtTiny(
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    best_val_accuracy = -1.0

    for epoch in range(NUM_EPOCHS):
        current_epoch = epoch + 1

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        print(
            f"Epoch [{current_epoch:02d}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        # Always keep the latest checkpoint.
        save_checkpoint(
            path=LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            epoch=current_epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )

        # Keep the checkpoint with the best validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                epoch=current_epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )

            print(
                f"  -> Best model saved "
                f"(validation accuracy: {val_accuracy:.4f})"
            )

    print("=" * 60)
    print("Training complete.")
    print(f"Best checkpoint: {BEST_MODEL_PATH}")
    print(f"Latest checkpoint: {LAST_MODEL_PATH}")
    print("=" * 60)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    train()
