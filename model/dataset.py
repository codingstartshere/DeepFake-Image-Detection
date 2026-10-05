"""
dataset.py

Defines CIFAKE preprocessing, PyTorch ImageFolder datasets, 
mDataLoaders, normalization, augmentation, and dataset information.
PyTorch dataset and DataLoader utilities for the CIFAKE
image deepfake detection project.

Dataset structure:

data/
├── train/
│   ├── FAKE/
│   └── REAL/
│
├── val/
│   ├── FAKE/
│   └── REAL/
│
└── test/
    ├── FAKE/
    └── REAL/

Classes:

    0 -> REAL
    1 -> FAKE
"""

from pathlib import Path

import torch  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader  # type: ignore[reportMissingImports]
from torchvision import datasets, transforms  # type: ignore[reportMissingImports]


# ============================================================
# 1. PATHS
# ============================================================

# Location of this file:
# deepfake-detection/model/dataset.py
MODEL_DIR = Path(__file__).resolve().parent

# Project root:
# deepfake-detection/
PROJECT_ROOT = MODEL_DIR.parent

# Dataset root:
# deepfake-detection/data/
DATASET_ROOT = PROJECT_ROOT / "data"

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

# CIFAKE images are 32 x 32.
IMAGE_SIZE = 32

# Number of images processed together.
DEFAULT_BATCH_SIZE = 128

# Number of parallel workers.
#
# 0 is safest on macOS, especially when initially testing
# the DataLoader.
DEFAULT_NUM_WORKERS = 0


# ============================================================
# 3. NORMALIZATION
# ============================================================

# CIFAKE is based on CIFAR-style 32x32 RGB images.
#
# These values are the commonly used CIFAR-10 channel
# statistics and provide a sensible starting point.
#
# We will NOT normalize the images differently between
# train, validation, and test.
CIFAR_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR_STD = (
    0.2470,
    0.2435,
    0.2616,
)


# ============================================================
# 4. TRANSFORMS
# ============================================================

# ------------------------------------------------------------
# Training transformations
# ------------------------------------------------------------
#
# We use only mild augmentation.
#
# This is important for deepfake detection because aggressive
# transformations may destroy the very image artifacts that
# the model needs to learn.
#
# We therefore avoid:
#   - RandomResizedCrop
#   - Heavy blur
#   - Heavy color distortion
#   - Random erasing
#
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=CIFAR_MEAN,
            std=CIFAR_STD,
        ),
    ]
)


# ------------------------------------------------------------
# Validation transformations
# ------------------------------------------------------------
#
# NO augmentation.
#
# Validation should represent the actual image distribution
# used to evaluate the model during training.
VAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Normalize(
            mean=CIFAR_MEAN,
            std=CIFAR_STD,
        ),
    ]
)


# ------------------------------------------------------------
# Test transformations
# ------------------------------------------------------------
#
# Test data must also remain untouched except for the
# deterministic preprocessing required by the model.
TEST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Normalize(
            mean=CIFAR_MEAN,
            std=CIFAR_STD,
        ),
    ]
)


# ============================================================
# 5. CREATE DATASETS
# ============================================================

def create_datasets():
    """
    Create the CIFAKE training, validation, and test datasets.

    Returns
    -------
    tuple
        train_dataset, val_dataset, test_dataset
    """

    # --------------------------------------------------------
    # Verify directories
    # --------------------------------------------------------

    required_directories = [
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
    ]

    for directory in required_directories:

        if not directory.exists():

            raise FileNotFoundError(
                f"Dataset directory does not exist:\n"
                f"{directory}"
            )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=TRAIN_TRANSFORM,
    )

    val_dataset = datasets.ImageFolder(
        root=str(VAL_DIR),
        transform=VAL_TRANSFORM,
    )

    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=TEST_TRANSFORM,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# 6. CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """
    Create DataLoaders for training, validation, and testing.

    Parameters
    ----------
    batch_size : int
        Number of images per batch.

    num_workers : int
        Number of worker processes used by DataLoader.

    Returns
    -------
    tuple
        train_loader, val_loader, test_loader
    """

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets()

    # --------------------------------------------------------
    # Training DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# 7. DATASET INFORMATION
# ============================================================

def get_dataset_information():
    """
    Return useful information about the datasets.

    Returns
    -------
    dict
        Dataset information.
    """

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets()

    return {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
    }


# ============================================================
# 8. MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CIFAKE DATASET MODULE")
    print("=" * 70)

    print()
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Image size   : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print()

    # Create datasets
    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets()

    print("DATASETS")
    print("-" * 70)

    print(
        f"Train      : {len(train_dataset)} images"
    )

    print(
        f"Validation : {len(val_dataset)} images"
    )

    print(
        f"Test       : {len(test_dataset)} images"
    )

    print()
    print("CLASSES")
    print("-" * 70)

    print(
        f"Classes       : {train_dataset.classes}"
    )

    print(
        f"Class mapping : {train_dataset.class_to_idx}"
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print()
    print("DATALOADERS")
    print("-" * 70)

    print(
        f"Train batches      : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(val_loader)}"
    )

    print(
        f"Test batches       : {len(test_loader)}"
    )

    # --------------------------------------------------------
    # Get one training batch
    # --------------------------------------------------------

    images, labels = next(iter(train_loader))

    print()
    print("SAMPLE TRAINING BATCH")
    print("-" * 70)

    print(
        f"Images shape : {images.shape}"
    )

    print(
        f"Labels shape : {labels.shape}"
    )

    print(
        f"Image dtype  : {images.dtype}"
    )

    print(
        f"Label dtype  : {labels.dtype}"
    )

    print()
    print(
        f"Image min    : {images.min().item():.4f}"
    )

    print(
        f"Image max    : {images.max().item():.4f}"
    )

    print()
    print("=" * 70)
    print("DATASET MODULE TEST COMPLETE")
    print("=" * 70)