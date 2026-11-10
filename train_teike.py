"""Reproducible TeiKe 70/15/15 training launcher.

The defaults are conservative for a 12 GB GPU: batch size 1, 320-pixel input,
five-frame clips, no DataLoader workers, CUDA expandable segments, and
fine-tuning from the released NUDT-MIRSDT checkpoint.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO / "Datasets" / "TeiKe_LMAFormer"
DEFAULT_CHECKPOINT = REPO / "result" / "NUDT-MIRSDT" / "checkpoint_NUDT-MIRSDT.pth"


def parse_args():
    parser = argparse.ArgumentParser("Train LMAFormer on the TeiKe split")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=320)
    parser.add_argument("--val-size", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="Optimizer steps in this run (0 = full epoch).")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-backbone", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_name = args.experiment_name or (
        "teike_split70_15_15_ft{}_ep{:02d}".format(args.train_size, args.epochs)
    )
    command = [
        sys.executable,
        str(REPO / "train_swin.py"),
        "--dataset_name",
        "TeiKe",
        "--output_dir",
        str(REPO / "result"),
        "--experiment_name",
        experiment_name,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        "1",
        "--train_size",
        str(args.train_size),
        "--val_size",
        str(args.val_size),
        "--num_frames",
        str(args.num_frames),
        "--num_workers",
        "0",
        "--max_train_steps",
        str(args.max_steps),
        "--lr",
        str(args.lr),
        "--lr_backbone",
        str(args.lr_backbone),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
    ]
    if args.amp:
        command.append("--amp")
    if args.init_checkpoint and args.init_checkpoint.is_file():
        command.extend(
            [
                "--pretrain_settings",
                "pretrained_model_path",
                str(args.init_checkpoint.resolve()),
            ]
        )

    environment = os.environ.copy()
    environment["LMAFORMER_DATASET_ROOT"] = str(args.data_root.resolve())
    environment["LMAFORMER_VAL_SPLIT"] = "val"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("dataset root:", environment["LMAFORMER_DATASET_ROOT"])
    print("train split:", args.data_root / "ImageSets" / "train_new.txt")
    print("val split:", args.data_root / "ImageSets" / "val_new.txt")
    print("command:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=REPO, env=environment)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
