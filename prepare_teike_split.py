"""Prepare the TeiKe 70/15/15 split for the LMAFormer dataloaders.

The source dataset stores frames as ``<root>/<sequence>/{images,masks}`` while
LMAFormer expects ``<dataset>/images/<sequence>``. This script creates directory
links without duplicating images and writes explicit train, val, and test
sequence lists from ``split_70_15_15.txt``.

Example::

    python prepare_teike_split.py
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from project_paths import LMADATASET_ROOT, TEIKE_DATA_ROOT, create_directory_link

DEFAULT_SOURCE = TEIKE_DATA_ROOT
DEFAULT_DESTINATION = LMADATASET_ROOT


def parse_args():
    parser = argparse.ArgumentParser("Prepare TeiKe for LMAFormer")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="TSV split file (default: <source>/split_70_15_15.txt)",
    )
    return parser.parse_args()


def read_split(path):
    sequences = {"train": [], "val": [], "test": []}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] == "split":
                continue
            if len(fields) < 2 or fields[0] not in sequences:
                raise ValueError("invalid split row {}: {!r}".format(line_number, line))
            sequences[fields[0]].append(fields[1])

    counts = Counter(name for names in sequences.values() for name in names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("sequences occur in more than one split: {}".format(duplicates))
    return sequences


def main():
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    split_path = args.split or source / "split_70_15_15.txt"
    sequences = read_split(split_path)

    expected = {"train": 70, "val": 15, "test": 15}
    actual = {name: len(names) for name, names in sequences.items()}
    if actual != expected:
        raise ValueError("expected split counts {}, got {}".format(expected, actual))

    all_sequences = [name for names in sequences.values() for name in names]
    missing = []
    for name in all_sequences:
        for kind in ("images", "masks"):
            if not (source / name / kind).is_dir():
                missing.append("{}/{}".format(name, kind))
    if missing:
        raise FileNotFoundError("missing source folders: {}".format(", ".join(missing)))

    created = 0
    for name in all_sequences:
        for kind in ("images", "masks"):
            created += create_directory_link(
                source / name / kind,
                destination / kind / name,
            )

    image_sets = destination / "ImageSets"
    image_sets.mkdir(parents=True, exist_ok=True)
    for split_name, names in sequences.items():
        output = image_sets / "{}_new.txt".format(split_name)
        output.write_text("\n".join(names) + "\n", encoding="utf-8")
        print("{}: {} sequences -> {}".format(split_name, len(names), output))

    for name in all_sequences:
        image_count = len(list((source / name / "images").glob("*.png")))
        mask_count = len(list((source / name / "masks").glob("*.png")))
        if image_count != 100 or mask_count != 100:
            raise ValueError(
                "sequence {} has images={}, masks={}; expected 100/100".format(
                    name, image_count, mask_count
                )
            )

    print("created {} directory links; validated images and masks for all sequences".format(created))
    return 0


if __name__ == "__main__":
    sys.exit(main())
