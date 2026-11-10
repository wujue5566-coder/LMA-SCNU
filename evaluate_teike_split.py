"""Evaluate a trained LMAFormer checkpoint on one TeiKe split."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from project_paths import LMADATASET_ROOT, TEIKE_DATA_ROOT

REPO = Path(__file__).resolve().parent
DEFAULT_SOURCE = TEIKE_DATA_ROOT
DEFAULT_SPLIT_ROOT = LMADATASET_ROOT / "ImageSets"


def parse_args():
    parser = argparse.ArgumentParser("Evaluate LMAFormer on a TeiKe split")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--val-size", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--msc", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_sequences(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main():
    args = parse_args()
    model_path = args.model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    sequences = read_sequences(args.split_root / "{}_new.txt".format(args.split))
    if args.max_seqs > 0:
        sequences = sequences[:args.max_seqs]
    out_root = (args.out_root or REPO / "predict" / "TeiKe_split70_15_15" / args.split).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["LMAFORMER_DATASET_ROOT"] = str(
        REPO / "Datasets" / "TeiKe_LMAFormer"
    )
    environment["LMAFORMER_VAL_SPLIT"] = args.split
    for sequence in sequences:
        metrics_path = out_root / sequence / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print("[skip] {}: {}".format(sequence, metrics_path), flush=True)
            continue
        command = [
            sys.executable,
            str(REPO / "eval_keii_masked.py"),
            "--sequence",
            sequence,
            "--data_dir",
            str(args.source / sequence),
            "--model_path",
            str(model_path),
            "--out_root",
            str(out_root),
            "--val_size",
            str(args.val_size),
            "--num_frames",
            str(args.num_frames),
            "--num_workers",
            "0",
            "--fps",
            "10",
            "--require_masks",
            "--no_plots",
        ]
        if not args.no_flip:
            command.append("--flip")
        if args.msc:
            command.append("--msc")
        if args.max_frames > 0:
            command.extend(["--max_frames", str(args.max_frames)])
        print("[eval {}] {}".format(args.split, sequence), flush=True)
        result = subprocess.run(command, cwd=REPO, env=environment)
        if result.returncode != 0:
            raise SystemExit("evaluation failed for sequence {} ({})".format(
                sequence, result.returncode))

    report_command = [
        sys.executable,
        str(REPO / "report_teike_eval.py"),
        "--run_root",
        str(out_root),
        "--data_root",
        str(args.source),
        "--output",
        str(out_root / "dataset_summary.png"),
        "--title",
        "LMAFormer fine-tuned on TeiKe / {} split".format(args.split),
    ]
    result = subprocess.run(report_command, cwd=REPO, env=environment)
    if result.returncode != 0:
        raise SystemExit("pooled report failed ({})".format(result.returncode))

    metrics = [json.loads((out_root / sequence / "metrics.json").read_text())
               for sequence in sequences
               if (out_root / sequence / "metrics.json").is_file()]
    summary = {
        "split": args.split,
        "sequences": len(metrics),
        "frames": sum(row["frames"] for row in metrics),
        "checkpoint": str(model_path),
        "val_size": args.val_size,
        "num_frames": args.num_frames,
        "flip": not args.no_flip,
        "msc": args.msc,
    }
    (out_root / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("summary: {}".format(out_root / "metrics_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
