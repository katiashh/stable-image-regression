# -*- coding: utf-8 -*-
"""
Rebuild labels.csv for an existing age_split directory.

Expected structure:

age_split/
    train/images/
    train/labels.csv
    val/images/
    val/labels.csv
    test/images/
    test/labels.csv

The script scans the actual image folders and writes labels.csv containing
only files that really exist and can be read. Existing ages are preserved.
If an image is not present in the old CSV, the age is reconstructed from an
IMDB-WIKI filename such as:

    36856076_1924-10-17_1958.jpg

using age = photo_year - birth_year.
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AGE_PATTERN = re.compile(r"(?P<birth_year>\d{4})-\d{2}-\d{2}[_-](?P<photo_year>\d{4})")


def normalize_filename(value):
    return Path(str(value).replace("\\", "/")).as_posix()


def parse_age_from_filename(filename):
    match = AGE_PATTERN.search(Path(filename).name)
    if match is None:
        return None

    birth_year = int(match.group("birth_year"))
    photo_year = int(match.group("photo_year"))
    age = photo_year - birth_year

    if 0 <= age <= 100:
        return float(age)
    return None


def read_old_labels(csv_path):
    if not csv_path.exists():
        return {}, {}

    df = pd.read_csv(csv_path)
    required_columns = {"filename", "age"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} must contain columns filename and age; "
            f"missing: {sorted(missing_columns)}"
        )

    exact = {}
    basename = {}

    for _, row in df.iterrows():
        filename = normalize_filename(row["filename"])
        try:
            age = float(row["age"])
        except (TypeError, ValueError):
            continue

        if not 0 <= age <= 100:
            continue

        exact[filename] = age
        basename.setdefault(Path(filename).name, age)

    return exact, basename


def collect_images(image_dir):
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    files = []
    unreadable = []

    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Verify that the file can actually be opened by OpenCV.
        if cv2.imread(str(path)) is None:
            unreadable.append(path)
            continue

        files.append(path)

    return files, unreadable


def rebuild_split(base_dir, split, overwrite=True):
    split_dir = base_dir / split
    image_dir = split_dir / "images"
    csv_path = split_dir / "labels.csv"

    exact_labels, basename_labels = read_old_labels(csv_path)
    image_files, unreadable = collect_images(image_dir)

    rows = []
    reconstructed = 0
    no_label = []

    for image_path in image_files:
        filename = image_path.relative_to(image_dir).as_posix()

        age = exact_labels.get(filename)
        if age is None:
            age = basename_labels.get(image_path.name)

        if age is None:
            age = parse_age_from_filename(image_path.name)
            if age is not None:
                reconstructed += 1

        if age is None:
            no_label.append(filename)
            continue

        rows.append({"filename": filename, "age": age})

    new_df = pd.DataFrame(rows, columns=["filename", "age"])
    new_df = new_df.sort_values("filename").reset_index(drop=True)

    if overwrite:
        if csv_path.exists():
            backup_path = split_dir / "labels_before_rebuild.csv"
            shutil.copy2(csv_path, backup_path)
        new_df.to_csv(csv_path, index=False)
        output_path = csv_path
    else:
        output_path = split_dir / "labels_rebuilt.csv"
        new_df.to_csv(output_path, index=False)

    print(f"\n[{split}]")
    print(f"  readable images found: {len(image_files)}")
    print(f"  unreadable images skipped: {len(unreadable)}")
    print(f"  rows without age skipped: {len(no_label)}")
    print(f"  ages reconstructed from filename: {reconstructed}")
    print(f"  final labels: {len(new_df)}")
    print(f"  saved to: {output_path}")

    if unreadable:
        print("  first unreadable files:")
        for path in unreadable[:10]:
            print(f"    - {path}")

    if no_label:
        print("  first files without age:")
        for filename in no_label[:10]:
            print(f"    - {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="data/wiki_crop/age_split",
        help="Path containing train, val, and test directories",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Write labels_rebuilt.csv instead of replacing labels.csv",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    print(f"Rebuilding split lists in: {base_dir.resolve()}")

    for split in ("train", "val", "test"):
        rebuild_split(
            base_dir=base_dir,
            split=split,
            overwrite=not args.no_overwrite,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
