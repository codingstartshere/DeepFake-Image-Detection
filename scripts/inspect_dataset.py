"""
inspect_dataset.py

Inspect the CIFAKE dataset used by the image deepfake detector.

Expected project structure:

deepfake-detection/
│
├── data/
│   ├── train/
│   │   ├── FAKE/
│   │   └── REAL/
│   │
│   └── test/
│       ├── FAKE/
│       └── REAL/
│
└── scripts/
    └── inspect_dataset.py

The script reports:
- Number of REAL and FAKE images
- Total images
- Image formats
- Image dimensions
- Corrupted/unreadable images
- Class balance
"""

from pathlib import Path
from collections import Counter

try:
    from PIL import Image  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "Pillow is required to inspect the CIFAKE dataset. "
        "Install it with `pip install pillow`."
    ) from exc


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Location of this file:
# deepfake-detection/scripts/inspect_dataset.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root:
# deepfake-detection/
PROJECT_ROOT = SCRIPT_DIR.parent

# CIFAKE dataset:
# deepfake-detection/data/
DATASET_ROOT = PROJECT_ROOT / "data"


# Dataset splits
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"


# Classes
CLASSES = ["REAL", "FAKE"]


# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_image_files(directory: Path):
    """
    Return all supported image files inside a directory.

    Parameters
    ----------
    directory : Path
        Directory containing images.

    Returns
    -------
    list[Path]
        List of image file paths.
    """

    if not directory.exists():
        return []

    return sorted(
        file
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_class(class_dir: Path):
    """
    Inspect all images belonging to one class.

    Parameters
    ----------
    class_dir : Path
        Directory containing images for one class.

    Returns
    -------
    dict
        Inspection information.
    """

    image_files = get_image_files(class_dir)

    format_counter = Counter()
    dimension_counter = Counter()

    corrupted_files = []

    total_pixels = 0

    for image_path in image_files:

        # Count file extension
        extension = image_path.suffix.lower()
        format_counter[extension] += 1

        try:
            with Image.open(image_path) as image:

                # Verify image integrity
                image.verify()

            # Re-open image after verify()
            with Image.open(image_path) as image:

                width, height = image.size

                dimension_counter[(width, height)] += 1

                total_pixels += width * height

        except Exception:
            corrupted_files.append(image_path)

    return {
        "count": len(image_files),
        "formats": format_counter,
        "dimensions": dimension_counter,
        "corrupted": corrupted_files,
        "total_pixels": total_pixels,
    }


def print_separator():
    """Print a visual separator."""

    print("=" * 70)


def print_class_statistics(split_name: str, split_dir: Path):
    """
    Print statistics for one dataset split.

    Parameters
    ----------
    split_name : str
        Name of the split, e.g. TRAIN or TEST.

    split_dir : Path
        Directory containing the split.
    """

    print()
    print_separator()
    print(f"{split_name} DATASET")
    print_separator()

    if not split_dir.exists():
        print(f"ERROR: Directory does not exist:")
        print(f"      {split_dir}")
        return

    split_total = 0
    split_corrupted = 0

    all_formats = Counter()
    all_dimensions = Counter()

    class_counts = {}

    for class_name in CLASSES:

        class_dir = split_dir / class_name

        print()
        print(f"{class_name}")
        print("-" * 70)

        if not class_dir.exists():
            print(f"WARNING: Directory does not exist:")
            print(f"         {class_dir}")

            class_counts[class_name] = 0
            continue

        statistics = inspect_class(class_dir)

        count = statistics["count"]
        corrupted = statistics["corrupted"]

        class_counts[class_name] = count

        split_total += count
        split_corrupted += len(corrupted)

        all_formats.update(statistics["formats"])
        all_dimensions.update(statistics["dimensions"])

        print(f"Number of images : {count}")
        print(f"Corrupted images : {len(corrupted)}")

        # ----------------------------------------------------
        # Image formats
        # ----------------------------------------------------

        print()
        print("Image formats:")

        if statistics["formats"]:

            for extension, number in statistics["formats"].most_common():
                print(f"  {extension:8s} : {number}")

        else:
            print("  No images found.")

        # ----------------------------------------------------
        # Image dimensions
        # ----------------------------------------------------

        print()
        print("Image dimensions:")

        if statistics["dimensions"]:

            for (width, height), number in statistics[
                "dimensions"
            ].most_common(10):

                print(
                    f"  {width} x {height}"
                    f" : {number}"
                )

            if len(statistics["dimensions"]) > 10:
                remaining = (
                    len(statistics["dimensions"]) - 10
                )

                print(
                    f"  ... and {remaining} other dimension(s)"
                )

        else:
            print("  No valid images found.")

        # ----------------------------------------------------
        # Corrupted files
        # ----------------------------------------------------

        if corrupted:

            print()
            print("Corrupted/unreadable files:")

            for file_path in corrupted[:10]:
                print(f"  {file_path}")

            if len(corrupted) > 10:
                print(
                    f"  ... and "
                    f"{len(corrupted) - 10} more."
                )

    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    print()
    print("-" * 70)
    print(f"{split_name} SUMMARY")
    print("-" * 70)

    real_count = class_counts.get("REAL", 0)
    fake_count = class_counts.get("FAKE", 0)

    print(f"REAL images       : {real_count}")
    print(f"FAKE images       : {fake_count}")
    print(f"Total images      : {split_total}")
    print(f"Corrupted images  : {split_corrupted}")

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    if split_total > 0:

        real_percentage = (
            real_count / split_total
        ) * 100

        fake_percentage = (
            fake_count / split_total
        ) * 100

        print()
        print("Class distribution:")

        print(
            f"  REAL : {real_percentage:.2f}%"
        )

        print(
            f"  FAKE : {fake_percentage:.2f}%"
        )

        # Difference between classes
        difference = abs(real_count - fake_count)

        if difference == 0:

            print()
            print("Class balance: PERFECTLY BALANCED")

        else:

            smaller = min(real_count, fake_count)

            if smaller > 0:

                imbalance_ratio = (
                    max(real_count, fake_count)
                    / smaller
                )

                print()
                print(
                    f"Class imbalance ratio: "
                    f"{imbalance_ratio:.2f}:1"
                )

                if imbalance_ratio <= 1.10:
                    print(
                        "Class balance: WELL BALANCED"
                    )

                elif imbalance_ratio <= 1.25:
                    print(
                        "Class balance: SLIGHTLY IMBALANCED"
                    )

                else:
                    print(
                        "Class balance: IMBALANCED"
                    )

    # --------------------------------------------------------
    # Overall formats
    # --------------------------------------------------------

    print()
    print("All image formats:")

    for extension, number in all_formats.most_common():
        print(f"  {extension:8s} : {number}")

    # --------------------------------------------------------
    # Overall dimensions
    # --------------------------------------------------------

    print()
    print("Most common image dimensions:")

    for (width, height), number in all_dimensions.most_common(10):

        print(
            f"  {width} x {height}"
            f" : {number}"
        )

    return {
        "REAL": real_count,
        "FAKE": fake_count,
        "TOTAL": split_total,
        "CORRUPTED": split_corrupted,
    }


# ============================================================
# 3. MAIN FUNCTION
# ============================================================

def main():

    print()
    print_separator()
    print("CIFAKE DATASET INSPECTION")
    print_separator()

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {DATASET_ROOT}")

    # ========================================================
    # Verify dataset root
    # ========================================================

    if not DATASET_ROOT.exists():

        print()
        print("ERROR: Dataset directory does not exist.")
        print()
        print(
            "Expected location:"
        )
        print(
            f"  {DATASET_ROOT}"
        )

        return

    # ========================================================
    # Inspect TRAIN
    # ========================================================

    train_statistics = print_class_statistics(
        "TRAIN",
        TRAIN_DIR,
    )

    # ========================================================
    # Inspect TEST
    # ========================================================

    test_statistics = print_class_statistics(
        "TEST",
        TEST_DIR,
    )

    # ========================================================
    # COMPLETE DATASET SUMMARY
    # ========================================================

    print()
    print_separator()
    print("COMPLETE DATASET SUMMARY")
    print_separator()

    if train_statistics and test_statistics:

        total_real = (
            train_statistics["REAL"]
            + test_statistics["REAL"]
        )

        total_fake = (
            train_statistics["FAKE"]
            + test_statistics["FAKE"]
        )

        total_images = (
            train_statistics["TOTAL"]
            + test_statistics["TOTAL"]
        )

        total_corrupted = (
            train_statistics["CORRUPTED"]
            + test_statistics["CORRUPTED"]
        )

        print()
        print(f"TRAIN images       : {train_statistics['TOTAL']}")
        print(f"TEST images        : {test_statistics['TOTAL']}")
        print(f"TOTAL images       : {total_images}")

        print()

        print(f"TRAIN REAL         : {train_statistics['REAL']}")
        print(f"TRAIN FAKE         : {train_statistics['FAKE']}")

        print()

        print(f"TEST REAL          : {test_statistics['REAL']}")
        print(f"TEST FAKE          : {test_statistics['FAKE']}")

        print()

        print(f"TOTAL REAL         : {total_real}")
        print(f"TOTAL FAKE         : {total_fake}")

        print()

        print(
            f"TOTAL CORRUPTED    : {total_corrupted}"
        )

        # ----------------------------------------------------
        # Overall class balance
        # ----------------------------------------------------

        if total_images > 0:

            real_percentage = (
                total_real / total_images
            ) * 100

            fake_percentage = (
                total_fake / total_images
            ) * 100

            print()
            print("OVERALL CLASS DISTRIBUTION")
            print("-" * 70)

            print(
                f"REAL : {real_percentage:.2f}%"
            )

            print(
                f"FAKE : {fake_percentage:.2f}%"
            )

    # ========================================================
    # Final message
    # ========================================================

    print()
    print_separator()
    print("DATASET INSPECTION COMPLETE")
    print_separator()
    print()


# ============================================================
# 4. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()