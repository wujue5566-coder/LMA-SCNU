"""Pool the per-sequence evaluation of a run into a dataset-level report.

Reads ``<run_root>/<sequence>/`` (``metrics.json`` + the saved 8-bit probability
maps under ``logits/``) together with the ground-truth masks, prints the
per-sequence table plus the pooled Pd / Fa / AUC, and draws a summary figure.

The pooled AUC is computed from the saved 8-bit probability maps, so it can
differ in the third decimal from the per-sequence value printed by
``eval_keii_masked.py`` (which uses the full-precision probabilities).

Example::

    python report_teike_eval.py --run_root ./predict/TeiKe_NUDT \
        --data_root ./Datasets/TeiKe/data \
        --output ./predict/TeiKe_NUDT/dataset_summary.png
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser("Dataset-level evaluation report")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--data_root", required=True,
                        help="Root with <sequence>/{images,masks}.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--title", default="LMAFormer / NUDT-MIRSDT on TeiKe "
                                         "(100 frames per sequence)")
    return parser.parse_args()


def read_metrics(path):
    with open(path) as handle:
        return json.load(handle)


def pool(run_root, data_root, threshold):
    from sklearn.metrics import roc_auc_score

    rows = []
    matched = targets = false_pixels = total_pixels = 0
    scores, labels = [], []
    for sequence_dir in sorted(run_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        metrics_path = sequence_dir / "metrics.json"
        logits_dir = sequence_dir / "logits"
        if not metrics_path.is_file() or not logits_dir.is_dir():
            continue
        metrics = read_metrics(metrics_path)
        rows.append(metrics)
        matched += metrics["detected_targets"]
        targets += metrics["total_targets"]
        false_pixels += metrics["false_alarm_pixels"]
        total_pixels += metrics["total_pixels"]
        ground_truth_dir = Path(data_root) / sequence_dir.name / "masks"
        for probability_path in sorted(logits_dir.glob("*.png")):
            mask_path = ground_truth_dir / probability_path.name
            if not mask_path.is_file():
                continue
            probability = cv2.imread(str(probability_path),
                                     cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            scores.append(probability.reshape(-1))
            labels.append(mask.reshape(-1).astype(np.uint8))
    if not rows:
        raise SystemExit("no sequence results found under {}".format(run_root))
    pooled_auc = float(roc_auc_score(np.concatenate(labels), np.concatenate(scores)))
    return rows, {
        "matched": matched,
        "targets": targets,
        "Pd": matched / targets if targets else float("nan"),
        "false_pixels": false_pixels,
        "total_pixels": total_pixels,
        "Fa": false_pixels / total_pixels if total_pixels else float("nan"),
        "AUC": pooled_auc,
    }


def main():
    args = parse_args()
    rows, pooled = pool(Path(args.run_root), Path(args.data_root), args.threshold)

    header = ("序列", "帧数", "GT目标", "命中", "Pd", "Fa", "AUC", "IoU", "nIoU")
    print("{:<6}{:>6}{:>8}{:>7}{:>9}{:>11}{:>9}{:>8}{:>8}".format(*header))
    for row in rows:
        print("{:<6}{:>6}{:>8}{:>7}{:>9.3f}{:>11.2e}{:>9.3f}{:>8.3f}{:>8.3f}".format(
            row["sequence"], row["frames"], row["total_targets"],
            row["detected_targets"], row["Pd"], row["Fa"], row["AUC"],
            row["IoU"], row["nIoU"]))
    print("{:<6}{:>6}{:>8}{:>7}{:>9.3f}{:>11.2e}{:>9.3f}".format(
        "汇总", sum(row["frames"] for row in rows), pooled["targets"],
        pooled["matched"], pooled["Pd"], pooled["Fa"], pooled["AUC"]))
    mean_iou = float(np.mean([row["IoU"] for row in rows]))
    mean_niou = float(np.mean([row["nIoU"] for row in rows]))
    pooled["mean_sequence_IoU"] = mean_iou
    pooled["mean_sequence_nIoU"] = mean_niou
    print("mean sequence IoU: {:.4f}".format(mean_iou))
    print("mean sequence nIoU: {:.4f}".format(mean_niou))

    with open(Path(args.run_root) / "dataset_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "frames", "gt_targets", "matched", "Pd", "Fa",
                         "AUC", "IoU", "nIoU"])
        for row in rows:
            writer.writerow([row["sequence"], row["frames"], row["total_targets"],
                             row["detected_targets"], row["Pd"], row["Fa"],
                             row["AUC"], row["IoU"], row["nIoU"]])
        writer.writerow(["pooled", sum(row["frames"] for row in rows),
                         pooled["targets"], pooled["matched"], pooled["Pd"],
                         pooled["Fa"], pooled["AUC"], mean_iou, mean_niou])
    with open(Path(args.run_root) / "pooled_metrics.json", "w", encoding="utf-8") as handle:
        import json
        json.dump(pooled, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if not args.output:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["sequence"] for row in rows]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    series = (("Pd", [row["Pd"] for row in rows], pooled["Pd"], "tab:blue"),
              ("Fa", [row["Fa"] for row in rows], pooled["Fa"], "tab:orange"),
              ("AUC", [row["AUC"] for row in rows], pooled["AUC"], "tab:green"),
              ("IoU", [row["IoU"] for row in rows], None, "tab:red"))
    for axis, (name, values, pooled_value, color) in zip(axes, series):
        axis.bar(positions, values, color=color, alpha=0.85)
        if pooled_value is not None:
            axis.axhline(pooled_value, color="k", ls="--", lw=1.0,
                         label="pooled {:.3f}".format(pooled_value))
            axis.legend(fontsize=7)
        axis.set_title(name, fontsize=10)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_xlabel("sequence")
        axis.grid(alpha=0.3, axis="y")
        if name == "Fa":
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    figure.suptitle(args.title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print("figure: {}".format(args.output))


if __name__ == "__main__":
    main()
