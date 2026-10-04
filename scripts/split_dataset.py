"""
split_dataset.py

Create an 80/20 train-validation split from the CIFAKE training dataset.

Expected initial structure:

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
    └── split_dataset.py

After running:

deepfake-detection/
│
├── data/
│   ├── train/
│   │   ├── FAKE/
│   │   └── REAL/
│   │
│   ├── val/
│   │   ├── FAKE/
│   │   └── REAL/
│   │
│   └── test/
│       ├── FAKE/
│       └── REAL/
│
└── scripts/
    └── split_dataset.py

The test dataset is NEVER modified by this script.

Default split:
    80% -> train
    20% -> validation

For CIFAKE:
    Original train:
        REAL = 50,000
        FAKE = 50,000

    Final train:
        REAL = 40,000
        FAKE = 40,000

    Validation:
        REAL = 10,000
        FAKE = 10,000

The split is reproducible using random seed 42.
"""

from pathlib import Path
import argparse
import csv
import random
import shutil
import sys


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Location of this script:
# deepfake-detection/scripts/split_dataset.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root:
# deepfake-detection/
PROJECT_ROOT = SCRIPT_DIR.parent

# Dataset root:
# deepfake-detection/data/
DATASET_ROOT = PROJECT_ROOT / "data"

# Dataset directories
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
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
    Return all supported image files in a directory.

    Files are returned in sorted order so that the initial
    ordering is deterministic before random sampling.
    """

    if not directory.exists():
        return []

    return sorted(
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_separator():
    """Print a visual separator."""

    print("=" * 70)


def count_images(directory: Path):
    """
    Count supported images in a directory.
    """

    return len(get_image_files(directory))


def check_dataset_structure():
    """
    Verify that the expected CIFAKE directory structure exists.

    Returns
    -------
    bool
        True if the structure is valid.
    """

    print()
    print("Checking dataset structure...")
    print()

    # --------------------------------------------------------
    # Dataset root
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():
        print("ERROR: Dataset directory does not exist:")
        print(f"       {DATASET_ROOT}")
        return False

    # --------------------------------------------------------
    # Train directory
    # --------------------------------------------------------

    if not TRAIN_DIR.exists():
        print("ERROR: Training directory does not exist:")
        print(f"       {TRAIN_DIR}")
        return False

    # --------------------------------------------------------
    # Training classes
    # --------------------------------------------------------

    for class_name in CLASSES:

        class_dir = TRAIN_DIR / class_name

        if not class_dir.exists():
            print(
                f"ERROR: Training class directory does not exist:"
            )
            print(f"       {class_dir}")
            return False

        image_count = count_images(class_dir)

        if image_count == 0:
            print(
                f"ERROR: No images found in:"
            )
            print(f"       {class_dir}")
            return False

    # --------------------------------------------------------
    # Test directory
    # --------------------------------------------------------

    if not TEST_DIR.exists():
        print("WARNING: Test directory does not exist:")
        print(f"         {TEST_DIR}")
        print()
        print(
            "The script will still split the training set,"
        )
        print(
            "but verify your dataset structure before continuing."
        )

    else:

        for class_name in CLASSES:

            class_dir = TEST_DIR / class_name

            if not class_dir.exists():
                print(
                    f"WARNING: Test class directory does not exist:"
                )
                print(f"         {class_dir}")

    print("Dataset structure: OK")

    return True


def check_validation_directory():
    """
    Ensure that the validation directory does not already
    contain files.

    This prevents accidental duplication or overwriting.
    """

    if not VAL_DIR.exists():
        return True

    # Check whether anything exists inside val/
    existing_files = [
        path
        for path in VAL_DIR.rglob("*")
        if path.is_file()
    ]

    if existing_files:

        print()
        print("ERROR: Validation directory is not empty.")
        print()
        print(f"Validation directory:")
        print(f"  {VAL_DIR}")
        print()
        print(
            f"Existing files: {len(existing_files)}"
        )
        print()
        print(
            "The script will NOT modify the existing validation data."
        )
        print()
        print(
            "If you intentionally want to recreate the split,"
        )
        print(
            "remove the existing val/ directory manually first."
        )

        return False

    return True


def calculate_validation_count(total_count, validation_ratio):
    """
    Calculate the number of images that should be moved
    into validation.
    """

    return int(total_count * validation_ratio)


def select_validation_files(
    image_files,
    validation_count,
    seed,
):
    """
    Select validation images using a reproducible random sample.

    A separate random generator is used for each class so that
    REAL and FAKE remain independently balanced.
    """

    rng = random.Random(seed)

    # Work on a copy so the original list isn't modified.
    files = list(image_files)

    rng.shuffle(files)

    return files[:validation_count]


def create_validation_directories():
    """
    Create:

    data/val/REAL
    data/val/FAKE
    """

    for class_name in CLASSES:

        class_dir = VAL_DIR / class_name
        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


def write_manifest(records):
    """
    Write a CSV manifest describing every file moved.

    The manifest is stored at:

        data/val_split_manifest.csv
    """

    manifest_path = DATASET_ROOT / "val_split_manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "class",
                "original_path",
                "validation_path",
                "seed",
            ]
        )

        writer.writerows(records)

    return manifest_path


def move_files(selected_files, class_name):
    """
    Move selected files from train/<class> to val/<class>.

    Returns
    -------
    list
        Manifest records for successfully moved files.
    """

    source_dir = TRAIN_DIR / class_name
    destination_dir = VAL_DIR / class_name

    records = []

    for source_path in selected_files:

        destination_path = (
            destination_dir / source_path.name
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if destination_path.exists():

            raise RuntimeError(
                "Destination file already exists:\n"
                f"  {destination_path}"
            )

        # ----------------------------------------------------
        # Move file
        # ----------------------------------------------------

        shutil.move(
            str(source_path),
            str(destination_path),
        )

        records.append(
            [
                class_name,
                str(source_path.relative_to(PROJECT_ROOT)),
                str(destination_path.relative_to(PROJECT_ROOT)),
                "",
            ]
        )

    return records


# ============================================================
# 3. MAIN SPLITTING FUNCTION
# ============================================================

def create_train_validation_split(
    validation_ratio=0.20,
    seed=42,
    dry_run=False,
):
    """
    Create the train/validation split.

    Parameters
    ----------
    validation_ratio : float
        Fraction of training data to move to validation.

    seed : int
        Random seed for reproducibility.

    dry_run : bool
        If True, do not move files. Only display what would happen.
    """

    print()
    print_separator()
    print("CIFAKE TRAIN / VALIDATION SPLIT")
    print_separator()

    print()
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Dataset root       : {DATASET_ROOT}")
    print(f"Training directory : {TRAIN_DIR}")
    print(f"Validation dir     : {VAL_DIR}")
    print(f"Test directory     : {TEST_DIR}")
    print()
    print(f"Validation ratio   : {validation_ratio:.0%}")
    print(f"Random seed        : {seed}")

    if dry_run:

        print()
        print("MODE                : DRY RUN")
        print(
            "No files will be moved."
        )

    else:

        print()
        print("MODE                : ACTUAL SPLIT")

    # ========================================================
    # CHECK DATASET STRUCTURE
    # ========================================================

    if not check_dataset_structure():
        return False

    # ========================================================
    # CHECK VALIDATION DIRECTORY
    # ========================================================

    if not check_validation_directory():
        return False

    # ========================================================
    # COLLECT FILES
    # ========================================================

    print()
    print_separator()
    print("SELECTING VALIDATION FILES")
    print_separator()

    selected_files_by_class = {}

    total_selected = 0

    for class_name in CLASSES:

        class_dir = TRAIN_DIR / class_name

        image_files = get_image_files(class_dir)

        total_images = len(image_files)

        validation_count = calculate_validation_count(
            total_images,
            validation_ratio,
        )

        selected_files = select_validation_files(
            image_files,
            validation_count,
            seed,
        )

        selected_files_by_class[class_name] = selected_files

        total_selected += len(selected_files)

        remaining_count = (
            total_images - len(selected_files)
        )

        print()
        print(f"{class_name}")
        print("-" * 70)
        print(f"Original images     : {total_images}")
        print(f"Validation images   : {len(selected_files)}")
        print(f"Remaining train     : {remaining_count}")

    # ========================================================
    # DISPLAY EXPECTED RESULT
    # ========================================================

    print()
    print_separator()
    print("EXPECTED RESULT")
    print_separator()

    total_original = 0

    for class_name in CLASSES:

        class_dir = TRAIN_DIR / class_name

        original_count = count_images(class_dir)

        validation_count = len(
            selected_files_by_class[class_name]
        )

        final_train_count = (
            original_count - validation_count
        )

        total_original += original_count

        print()
        print(f"{class_name}:")
        print(f"  Train      : {final_train_count}")
        print(f"  Validation : {validation_count}")

    print()
    print(f"Original training images : {total_original}")
    print(f"Validation images        : {total_selected}")
    print(
        f"Final training images   : "
        f"{total_original - total_selected}"
    )

    # ========================================================
    # DRY RUN ENDS HERE
    # ========================================================

    if dry_run:

        print()
        print_separator()
        print("DRY RUN COMPLETE")
        print_separator()

        print()
        print(
            "No files were moved."
        )

        return True

    # ========================================================
    # FINAL CONFIRMATION
    # ========================================================

    print()
    print_separator()
    print("READY TO CREATE SPLIT")
    print_separator()

    print()
    print(
        "The following operation will:"
    )

    print(
        "  1. Move 20% of each training class to data/val/"
    )

    print(
        "  2. Leave 80% of each class in data/train/"
    )

    print(
        "  3. NOT modify data/test/"
    )

    print(
        "  4. Create data/val_split_manifest.csv"
    )

    print()

    # ========================================================
    # CREATE VALIDATION DIRECTORIES
    # ========================================================

    create_validation_directories()

    # ========================================================
    # MOVE FILES
    # ========================================================

    all_manifest_records = []

    try:

        for class_name in CLASSES:

            print()
            print(
                f"Moving {class_name} images..."
            )

            selected_files = selected_files_by_class[
                class_name
            ]

            records = move_files(
                selected_files,
                class_name,
            )

            # Add seed to every record
            for record in records:
                record[3] = str(seed)

            all_manifest_records.extend(records)

            print(
                f"Moved {len(records)} "
                f"{class_name} images."
            )

    except Exception as error:

        print()
        print_separator()
        print("ERROR DURING SPLIT")
        print_separator()

        print()
        print(error)

        print()
        print(
            "Some files may already have been moved."
        )

        print(
            "The script will not automatically delete or"
        )
        print(
            "overwrite any files."
        )

        return False

    # ========================================================
    # WRITE MANIFEST
    # ========================================================

    manifest_path = write_manifest(
        all_manifest_records
    )

    # ========================================================
    # VERIFY FINAL COUNTS
    # ========================================================

    print()
    print_separator()
    print("VERIFYING RESULT")
    print_separator()

    verification_passed = True

    for class_name in CLASSES:

        train_count = count_images(
            TRAIN_DIR / class_name
        )

        val_count = count_images(
            VAL_DIR / class_name
        )

        print()
        print(f"{class_name}:")
        print(f"  Train      : {train_count}")
        print(f"  Validation : {val_count}")

        # Expected validation count
        original_count = (
            train_count + val_count
        )

        expected_validation = calculate_validation_count(
            original_count,
            validation_ratio,
        )

        if val_count != expected_validation:

            verification_passed = False

            print(
                "  WARNING: Validation count does not "
                "match expected value."
            )

    # ========================================================
    # VERIFY TEST DATA WAS NOT TOUCHED
    # ========================================================

    print()
    print("Checking test dataset...")

    if TEST_DIR.exists():

        for class_name in CLASSES:

            test_class_dir = (
                TEST_DIR / class_name
            )

            test_count = count_images(
                test_class_dir
            )

            print(
                f"  TEST {class_name}: "
                f"{test_count} images"
            )

    else:

        print(
            "  Test directory does not exist."
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print_separator()

    if verification_passed:

        print("SPLIT COMPLETED SUCCESSFULLY")

    else:

        print("SPLIT COMPLETED WITH WARNINGS")

    print_separator()

    print()
    print(
        f"Manifest: {manifest_path}"
    )

    print()

    print(
        "Final dataset structure:"
    )

    print()
    print(
        "data/"
    )
    print(
        "├── train/"
    )
    print(
        "│   ├── FAKE/"
    )
    print(
        "│   └── REAL/"
    )
    print(
        "│"
    )
    print(
        "├── val/"
    )
    print(
        "│   ├── FAKE/"
    )
    print(
        "│   └── REAL/"
    )
    print(
        "│"
    )
    print(
        "└── test/"
    )
    print(
        "    ├── FAKE/"
    )
    print(
        "    └── REAL/"
    )

    print()

    return verification_passed


# ============================================================
# 4. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Create an 80/20 train-validation split "
            "for the CIFAKE dataset."
        )
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.20,
        help=(
            "Fraction of training data used for validation "
            "(default: 0.20)"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducible splitting "
            "(default: 42)"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show what would happen without moving files."
        ),
    )

    return parser.parse_args()


# ============================================================
# 5. SCRIPT ENTRY POINT
# ============================================================

def main():

    args = parse_arguments()

    # --------------------------------------------------------
    # Validate ratio
    # --------------------------------------------------------

    if not 0 < args.validation_ratio < 1:

        print(
            "ERROR: validation ratio must be "
            "between 0 and 1."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    success = create_train_validation_split(
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()