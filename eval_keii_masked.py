"""Score LMAFormer on a Keii sequence that now has ground-truth masks.

`Datasets/Keii/<sequence>/` is expected to hold ``images/`` and ``masks/`` (the
layout used by the 100-frame Keii subsets).  Inference reuses the Keii video
pipeline from ``inference_keii.py``: 5-frame sliding window with wrap-around,
shortest side scaled to ``--val_size``, ImageNet normalisation, optional
flip/msc averaging, probability threshold 0.5.

Metrics follow the paper definitions: Pd = N_true / N_gt (targets detected with
the project's 3-pixel centroid rule), Fa = N_false / N_all (false-alarm pixels
over all pixels), and ROC / AUC from sweeping the probability threshold.

The ROC is the standard TPR-vs-FPR curve (vertical axis = detection rate,
horizontal axis = false-alarm rate) and the reported AUC is the area enclosed by
that curve and the axes, i.e. the probability that a target pixel scores above a
background pixel.  The target-count Pd against pixel-level Fa curve - which is
what Eq. 32/33 of the paper describes for the operating point - is kept in
``metrics.json`` (``AUC_pd_fa_logFa``) but is not the reported AUC, because below
Fa ~ 1e-4 the whole frame merges into one component and the centroid matching
rule collapses.

IoU / nIoU are reported as secondary mask-footprint numbers (they are dominated
by how tightly the ground-truth blob is drawn when targets span ~50 pixels).

Outputs: per-frame probability and binary masks, a three-panel comparison video
(ground truth | prediction | TP/FN/FP overlay), a summary figure with the three
metrics plus the ROC curve, and ``metrics.json`` / ``metrics.csv``.

Example::

    python eval_keii_masked.py --sequence 7 \
        --data_dir ./Datasets/Keii/7 \
        --model_path ./result/NUDT-MIRSDT/checkpoint_NUDT-MIRSDT.pth \
        --output_dir ./predict/Keii_NUDT_gt
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import visualize_irdst as viz
from inference_keii import build_model, predict_center, transform
from inference_swin import get_args_parser
from MFIRSTD.metrics import PD0_FA0, ROCMetric, SamplewiseSigmoidMetric, SigmoidMetric

try:  # numpy >= 2.0 renamed trapz
    trapezoid = np.trapezoid
except AttributeError:  # pragma: no cover - older numpy
    trapezoid = np.trapz


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate LMAFormer on a Keii sequence with ground-truth masks",
        parents=[get_args_parser()])
    parser.add_argument("--sequence", default="7",
                        help="Sequence name, used for the output subfolder and reporting.")
    parser.add_argument("--data_dir", default="./Datasets/Keii/7",
                        help="Folder holding images/ and masks/ for this sequence.")
    parser.add_argument("--images_dir", default=None,
                        help="Frame folder, overriding <data_dir>/images.")
    parser.add_argument("--masks_dir", default=None,
                        help="Mask folder, overriding <data_dir>/masks. Frames without a "
                             "mask are still predicted and rendered, they only stay out of "
                             "the metrics.")
    parser.add_argument("--require_masks", action="store_true",
                        help="Abort when a frame has no ground-truth mask.")
    parser.add_argument("--out_root", default="./predict/Keii_NUDT_gt",
                        help="Root for the evaluation outputs; the sequence folder "
                             "is created inside it.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=1,
                        help="Minimum component area for drawn boxes (metrics use all pixels).")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate of the output video.")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Limit the number of frames (0 = all).")
    parser.add_argument("--no_masks", action="store_true",
                        help="Do not write the per-frame probability / binary PNGs.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Do not write the ROC / Pd-Fa curve figure.")
    parser.add_argument("--sweep_bins", type=int, default=256,
                        help="Number of probability thresholds swept for the ROC/AUC.")
    return parser.parse_args()


def list_frames(images_dir, masks_dir, max_frames, require_masks=False):
    image_paths = sorted(images_dir.glob("*.png"), key=viz.natural_key)
    if not image_paths:
        raise FileNotFoundError("No frames in {}".format(images_dir))
    if max_frames > 0:
        image_paths = image_paths[:max_frames]
    pairs = []
    for image_path in image_paths:
        mask_path = masks_dir / image_path.name if masks_dir else None
        if mask_path is None or not mask_path.is_file():
            if require_masks:
                raise FileNotFoundError("No mask for {}".format(image_path.name))
            mask_path = None
        pairs.append((image_path, mask_path))
    return pairs


def resolve_dirs(args):
    """Frame / mask folders: explicit flags win, otherwise <data_dir>/{images,masks}."""
    if args.images_dir:
        images_dir = Path(args.images_dir)
        masks_dir = Path(args.masks_dir) if args.masks_dir else None
    else:
        data_dir = Path(args.data_dir)
        images_dir = data_dir / "images"
        masks_dir = data_dir / "masks"
    return images_dir, masks_dir


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Unable to read image: {}".format(path))
    return image


def window_indices(center, total, num_frames):
    half = num_frames // 2
    return [(center + offset - half) % total for offset in range(num_frames)]


def main():
    args = parse_args()
    if not hasattr(args, "max_sc") or args.max_sc is None:
        args.max_sc = 1.8
    args.aux_loss = 0
    args.aux_loss_norm = 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sequence_dir = Path(args.out_root) / args.sequence
    sequence_dir.mkdir(parents=True, exist_ok=True)

    images_dir, masks_dir = resolve_dirs(args)
    pairs = list_frames(images_dir, masks_dir, args.max_frames, args.require_masks)
    total = len(pairs)
    rgb_frames = [cv2.cvtColor(read_rgb(image_path), cv2.COLOR_BGR2RGB)
                  for image_path, _ in pairs]
    bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames]
    masks = [viz.read_mask(mask_path) if mask_path else None for _, mask_path in pairs]
    gt_indices = [index for index, mask in enumerate(masks) if mask is not None]
    if not gt_indices:
        raise FileNotFoundError("no ground-truth masks found in {}".format(masks_dir))
    print("sequence {}: {} frames, {} with ground truth | device {} | val_size {} | "
          "flip {} | msc {}".format(args.sequence, total, len(gt_indices), device,
                                    args.val_size, args.flip, args.msc))
    print("checkpoint: {}".format(args.model_path))

    model = build_model(args, device)
    scales = [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3] if args.msc else [1]
    gt_shape = masks[gt_indices[0]].shape

    probabilities = []
    started = time.time()
    for index in range(total):
        clip = [rgb_frames[frame] for frame in window_indices(index, total, args.num_frames)]
        tensors = transform(clip, args.val_size, args.max_sc)
        probability = predict_center(model, tensors, device, gt_shape, args.flip,
                                     args.threshold, scales)
        probabilities.append(probability)
    elapsed = time.time() - started
    print("inference: {} frames in {:.1f}s ({:.2f} s/frame)".format(
        total, elapsed, elapsed / max(total, 1)))

    binaries = [probability > args.threshold for probability in probabilities]

    pd_fa = PD0_FA0(nclass=1, thre=args.threshold)
    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(nclass=1)
    roc = ROCMetric(nclass=1, bins=255)
    frame_ious = [None] * total
    for index in gt_indices:
        probability, binary, ground_truth = probabilities[index], binaries[index], masks[index]
        pd_fa.update(probability, ground_truth)
        iou_metric.update(binary.astype(np.uint8), ground_truth.astype(np.uint8))
        niou_metric.update(binary.astype(np.uint8), ground_truth.astype(np.uint8))
        roc.update(probability.astype(np.float32), ground_truth.astype(np.uint8))
        frame_ious[index] = viz.iou_score(ground_truth, binary)

    fa, pd = pd_fa.get()
    _, iou, _ = iou_metric.get()
    niou = niou_metric.get()
    tp_rates, fp_rates, _, _ = roc.get()
    # ROCMetric sweeps threshold 0 -> 1, so the rates run from (1, 1) to (0, 0);
    # integrating against the reversed arrays gives the pixel-level ROC area.
    auc_pixel_bins = float(trapezoid(tp_rates[::-1], fp_rates[::-1]))
    scores = np.concatenate([probabilities[index].reshape(-1) for index in gt_indices])
    labels = np.concatenate([masks[index].reshape(-1).astype(np.uint8)
                             for index in gt_indices])
    auc_ranking = rank_auc(scores, labels)

    # Paper protocol: Pd is the *target* detection ratio, Fa the pixel false-alarm
    # ratio, and the ROC/AUC are read off the Pd-vs-Fa curve over thresholds.
    thresholds, pd_curve, fa_curve = sweep_pd_fa(
        [probabilities[index] for index in gt_indices],
        [masks[index] for index in gt_indices], args.sweep_bins)
    order = np.argsort(fa_curve)
    auc_linear = float(trapezoid(pd_curve[order], fa_curve[order]))
    auc_pd_fa = auc_log_fa(pd_curve, fa_curve)
    detected_targets = int(round(pd_fa.PD0))
    total_targets = int(pd_fa.target)

    metrics = {
        "sequence": args.sequence,
        "checkpoint": args.model_path,
        "frames": total,
        "gt_frames": len(gt_indices),
        "val_size": args.val_size,
        "flip": bool(args.flip),
        "msc": bool(args.msc),
        "threshold": args.threshold,
        "sweep_bins": args.sweep_bins,
        "Pd": float(pd),
        "Fa": float(fa),
        "AUC": auc_ranking,
        "AUC_roc_polygon_256": auc_pixel_bins,
        "AUC_pd_fa_logFa": auc_pd_fa,
        "AUC_linear_PdFa": auc_linear,
        "IoU": float(iou),
        "nIoU": float(niou),
        "detected_targets": detected_targets,
        "total_targets": total_targets,
        "false_alarm_pixels": int(pd_fa.FA0),
        "total_pixels": int(pd_fa.exlment_num),
        "mean_frame_iou": float(np.mean([value for value in frame_ious
                                         if value is not None])),
        "seconds": elapsed,
    }

    if not args.no_masks:
        logits_dir = sequence_dir / "logits"
        binary_dir = sequence_dir / "bin_mask"
        logits_dir.mkdir(exist_ok=True)
        binary_dir.mkdir(exist_ok=True)
        for (image_path, _), probability, binary in zip(pairs, probabilities, binaries):
            cv2.imwrite(str(logits_dir / image_path.name),
                        (probability * 255).astype(np.uint8))
            cv2.imwrite(str(binary_dir / image_path.name),
                        (binary.astype(np.uint8) * 255))

    write_metrics(sequence_dir, metrics)
    write_roc_csv(sequence_dir, thresholds, pd_curve, fa_curve)
    if not args.no_plots:
        write_summary_figure(sequence_dir, bgr_frames, masks, binaries, frame_ious,
                             tp_rates, fp_rates, metrics)
    write_video(sequence_dir, bgr_frames, masks, binaries, frame_ious, metrics,
                args.fps, args.min_area)

    for key in ("Pd", "Fa", "AUC", "IoU", "nIoU"):
        print("{}: {:.5f}".format(key, metrics[key]))
    print("targets matched {}/{} | false-alarm pixels {}/{}".format(
        detected_targets, total_targets, metrics["false_alarm_pixels"],
        metrics["total_pixels"]))
    print("outputs: {}".format(sequence_dir))


def write_metrics(sequence_dir, metrics):
    with open(sequence_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(sequence_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def match_targets(ground_truth, prediction, distance=3.0):
    """Replicate MFIRSTD.metrics.PD0_FA0's matching rule for one frame."""
    gt_count, _, _, gt_centroids = cv2.connectedComponentsWithStats(
        ground_truth.astype(np.uint8), connectivity=8)
    pred_count, _, _, pred_centroids = cv2.connectedComponentsWithStats(
        prediction.astype(np.uint8), connectivity=8)
    unmatched = list(range(1, pred_count))
    matched = 0
    for gt_id in range(1, gt_count):
        for position, pred_id in enumerate(unmatched):
            if np.linalg.norm(gt_centroids[gt_id] - pred_centroids[pred_id]) < distance:
                matched += 1
                del unmatched[position]
                break
    return matched, gt_count - 1


def auc_log_fa(pd_curve, fa_curve):
    """Area under the Pd-Fa ROC with a logarithmic Fa axis, normalised to [0, 1].

    Fa spans several decades (roughly 1e-7 .. 1e-4) while Pd moves only over the
    top of its range, so the linear-Fa area is dominated by a couple of samples
    near threshold 0.  Papers that quote an AUC for this Pd/Fa pair integrate the
    curve on the log axis, which is what this returns.
    """
    positive = fa_curve > 0
    if positive.sum() < 2:
        return float("nan")
    log_fa = np.log10(fa_curve[positive])
    pd = pd_curve[positive]
    order = np.argsort(log_fa)
    log_fa, pd = log_fa[order], pd[order]
    span = log_fa[-1] - log_fa[0]
    if span <= 0:
        return float("nan")
    return float(trapezoid(pd, (log_fa - log_fa[0]) / span))


def write_roc_csv(sequence_dir, thresholds, pd_curve, fa_curve):
    path = sequence_dir / "roc_curve.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "Pd", "Fa"])
        for threshold, pd, fa in zip(thresholds, pd_curve, fa_curve):
            writer.writerow(["{:.6f}".format(threshold), "{:.6f}".format(pd),
                             "{:.10e}".format(fa)])


def sweep_pd_fa(probabilities, masks, bins):
    """Pd (target ratio) and Fa (pixel ratio) for every swept threshold."""
    thresholds = np.linspace(0.0, 1.0, max(bins, 2))
    total_targets = sum(int(cv2.connectedComponents(mask.astype(np.uint8),
                                                    connectivity=8)[0]) - 1 for mask in masks)
    total_pixels = sum(int(mask.size) for mask in masks)
    pd_curve = np.zeros(len(thresholds))
    fa_curve = np.zeros(len(thresholds))
    for index, threshold in enumerate(thresholds):
        matched = false_pixels = 0
        for probability, mask in zip(probabilities, masks):
            prediction = probability > threshold
            hits, _ = match_targets(mask, prediction)
            matched += hits
            false_pixels += int(np.logical_and(prediction, ~mask).sum())
        pd_curve[index] = matched / total_targets if total_targets else 0.0
        fa_curve[index] = false_pixels / total_pixels if total_pixels else 0.0
    return thresholds, pd_curve, fa_curve


def draw_example(frame, ground_truth, prediction, min_area=1):
    """BGR overlay with the GT box (yellow), the prediction box (red) and an ROI."""
    view = frame.copy()
    layers = [(viz.connected_boxes(prediction, min_area), viz.PREDICTION_COLOR)]
    if ground_truth is not None:
        layers.insert(0, (viz.connected_boxes(ground_truth, min_area),
                          viz.GROUND_TRUTH_COLOR))
    for boxes, color in layers:
        for x, y, width, height, _ in boxes:
            size = max(width, height, 10)
            x1 = max(0, x - (size - width) // 2)
            y1 = max(0, y - (size - height) // 2)
            x2 = min(view.shape[1] - 1, x1 + size - 1)
            y2 = min(view.shape[0] - 1, y1 + size - 1)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    roi = zoom_box(ground_truth, prediction, view.shape)
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(view, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1, cv2.LINE_AA)
    return view


def zoom_box(ground_truth, prediction, shape, half=28):
    """ROI around the target (or the prediction when the target is missed)."""
    mask = ground_truth if ground_truth is not None and ground_truth.any() else prediction
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    height, width = shape[:2]
    center_x = int(round(xs.mean()))
    center_y = int(round(ys.mean()))
    x0 = max(0, min(width - 1, center_x - half))
    y0 = max(0, min(height - 1, center_y - half))
    x1 = min(width, x0 + 2 * half)
    y1 = min(height, y0 + 2 * half)
    return x0, y0, x1, y1


def write_summary_figure(sequence_dir, frames, masks, binaries, frame_ious,
                         tp_rates, fp_rates, metrics):
    """One image holding the three metrics, the ROC curve and two example frames."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched_frames = [index for index, value in enumerate(frame_ious)
                      if value is not None and value > 0]
    missed_frames = [index for index, value in enumerate(frame_ious) if value == 0]
    good = matched_frames[len(matched_frames) // 2] if matched_frames else 0
    bad = missed_frames[0] if missed_frames else good

    figure = plt.figure(figsize=(13.2, 5.6))
    grid = figure.add_gridspec(2, 3, height_ratios=(2.1, 1.0), hspace=0.32, wspace=0.22)
    for column, (index, title) in enumerate((
            (good, "detected: frame {}".format(good)),
            (bad, "missed: frame {}".format(bad)))):
        axes = figure.add_subplot(grid[0, column])
        example = draw_example(frames[index], masks[index], binaries[index])
        axes.imshow(cv2.cvtColor(example, cv2.COLOR_BGR2RGB))
        axes.set_title("{}\nGT yellow | prediction red | white = zoom box".format(title),
                       fontsize=9)
        axes.axis("off")
        roi = zoom_box(masks[index], binaries[index], example.shape)
        if roi is not None:
            x0, y0, x1, y1 = roi
            zoom_axes = axes.inset_axes((0.60, 0.04, 0.38, 0.42))
            zoom_axes.imshow(cv2.cvtColor(example[y0:y1, x0:x1], cv2.COLOR_BGR2RGB),
                             interpolation="nearest")
            zoom_axes.set_xticks([])
            zoom_axes.set_yticks([])
            for spine in zoom_axes.spines.values():
                spine.set_edgecolor("white")

    roc_axes = figure.add_subplot(grid[0, 2])
    # ROC = TPR (detection rate) versus FPR (false-alarm rate); AUC is the area
    # enclosed by the curve and the axes, so the curve is drawn on linear axes.
    roc_axes.plot([0, 1], [0, 1], ls=":", lw=1.0, color="grey")
    roc_axes.plot(fp_rates, tp_rates, lw=1.8, color="tab:blue")
    roc_axes.fill_between(fp_rates, tp_rates, color="tab:blue", alpha=0.15)
    roc_axes.set_xlim(-0.02, 1.02)
    roc_axes.set_ylim(-0.02, 1.02)
    roc_axes.set_xlabel("FPR (false-alarm rate)")
    roc_axes.set_ylabel("TPR (detection rate)")
    roc_axes.set_title("ROC  AUC = {:.4f}".format(metrics["AUC"]))
    roc_axes.grid(alpha=0.3)
    # Zoom on the low-FPR corner where the operating point sits.
    zoom_axes = roc_axes.inset_axes((0.42, 0.14, 0.54, 0.5))
    zoom_axes.plot(fp_rates, tp_rates, lw=1.4, color="tab:blue")
    zoom_axes.fill_between(fp_rates, tp_rates, color="tab:blue", alpha=0.15)
    zoom_axes.scatter([metrics["Fa"]], [metrics["Pd"]], s=22, zorder=5,
                      color="tab:red")
    zoom_axes.set_xscale("log")
    zoom_axes.set_xlim(1e-7, 1e-3)
    zoom_axes.set_ylim(0.6, 1.02)
    zoom_axes.set_title("zoom: threshold {:.2f}".format(metrics["threshold"]),
                        fontsize=7)
    zoom_axes.tick_params(labelsize=6)
    zoom_axes.grid(alpha=0.3)

    text_axes = figure.add_subplot(grid[1, :])
    text_axes.axis("off")
    text_axes.text(0.0, 1.0, (
        "Pd = N_true / N_gt = {matched}/{total} = {pd:.4f}\n"
        "Fa = N_false / N_all = {false}/{pixels} = {fa:.2e}\n"
        "AUC = area under the TPR-FPR ROC curve = {auc:.4f}"
        "   [256-point polygon {poly:.4f}]\n"
        "threshold = {threshold:.2f} | NUDT-MIRSDT | Keii seq {sequence} | {frames} frames | "
        "GT for {gt} of them | flip = {flip} | msc = {msc}").format(
            matched=metrics["detected_targets"], total=metrics["total_targets"],
            pd=metrics["Pd"], false=metrics["false_alarm_pixels"],
            pixels=metrics["total_pixels"], fa=metrics["Fa"], auc=metrics["AUC"],
            poly=metrics["AUC_roc_polygon_256"],
            threshold=metrics["threshold"], sequence=metrics["sequence"],
            frames=metrics["frames"], gt=metrics["gt_frames"],
            flip=metrics["flip"], msc=metrics["msc"]),
        va="top", ha="left", family="DejaVu Sans Mono", fontsize=11)

    path = sequence_dir / "metrics_summary.png"
    save_figure(figure, path)
    plt.close(figure)
    print("summary figure: {}".format(path))


def save_figure(figure, path, dpi=150):
    """Write the figure through a temporary file so an open viewer cannot block us."""
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(str(temporary), dpi=dpi)
    for attempt in range(5):
        try:
            os.replace(str(temporary), str(path))
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5)


def rank_auc(scores, labels):
    """Tie-aware pixel-level ROC AUC (equals sklearn's roc_auc_score).

    The 256-step threshold sweep used for the Pd/Fa curve is too coarse to
    integrate the ROC directly, so the reported AUC is the probability that a
    random target pixel scores above a random background pixel.
    """
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except ImportError:  # pragma: no cover - sklearn is part of the env
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
        sorted_scores = scores[order]
        start = 0
        for index in range(1, len(sorted_scores) + 1):
            if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
                if index - start > 1:
                    ranks[order[start:index]] = ranks[order[start:index]].mean()
                start = index
        positives = labels > 0
        n_pos = int(positives.sum())
        n_neg = int(len(labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def write_video(sequence_dir, frames, masks, binaries, frame_ious, metrics, fps, min_area):
    """Three panels per frame: ground truth | prediction | TP/FN/FP overlay."""
    height, width = frames[0].shape[:2]
    footer_height = 48
    output_path = sequence_dir / "overlay.mp4"
    preview_path = sequence_dir / "overlay_first_frame.png"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width * 3, height + footer_height))
    if not writer.isOpened():
        raise RuntimeError("Unable to open writer for {}".format(output_path))

    header = ("Pd {:.3f} | Fa {:.2e} | AUC {:.4f} (TPR-FPR ROC)".format(
        metrics["Pd"], metrics["Fa"], metrics["AUC"]))
    try:
        for index, (image, ground_truth, prediction) in enumerate(
                zip(frames, masks, binaries)):
            prediction_boxes = viz.connected_boxes(prediction, min_area)

            prediction_panel = viz.overlay_mask(image, prediction, viz.PREDICTION_COLOR)
            viz.draw_boxes(prediction_panel, prediction_boxes,
                           viz.PREDICTION_COLOR, "P-")
            viz.put_label(prediction_panel, "Prediction", (10, 27),
                          viz.PREDICTION_COLOR)

            if ground_truth is None:
                # No annotation for this frame: show the plain frame and mark it.
                ground_truth_panel = image.copy()
                combined_panel = viz.overlay_mask(image, prediction,
                                                  viz.PREDICTION_COLOR)
                viz.draw_boxes(combined_panel, prediction_boxes,
                               viz.PREDICTION_COLOR, "P-")
                viz.put_label(ground_truth_panel, "no ground truth", (10, 27),
                              viz.TEXT_COLOR)
                viz.put_label(combined_panel, "Prediction only", (10, 27),
                              viz.PREDICTION_COLOR)
            else:
                ground_truth_boxes = viz.connected_boxes(ground_truth, min_area)
                matched_boxes, false_alarm_boxes = viz.split_prediction_boxes(
                    ground_truth, prediction, min_area)
                ground_truth_panel = viz.overlay_mask(image, ground_truth,
                                                      viz.GROUND_TRUTH_COLOR)
                combined_panel = viz.overlay_masks(image, ground_truth, prediction)
                viz.draw_boxes(ground_truth_panel, ground_truth_boxes,
                               viz.GROUND_TRUTH_COLOR, "GT-")
                viz.draw_boxes(combined_panel, ground_truth_boxes,
                               viz.MISSED_TARGET_COLOR, "GT-")
                viz.draw_boxes(combined_panel, matched_boxes, viz.PREDICTION_COLOR, "P-")
                viz.draw_boxes(combined_panel, false_alarm_boxes,
                               viz.FALSE_ALARM_COLOR, "F-")
                viz.put_label(ground_truth_panel, "Ground truth", (10, 27),
                              viz.GROUND_TRUTH_COLOR)
                viz.draw_overlay_legend(combined_panel)

            footer = np.full((footer_height, width * 3, 3), viz.PANEL_COLOR, dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(footer, "Frame {}/{}".format(index + 1, len(frames)), (8, 18),
                        font, 0.43, viz.TEXT_COLOR, 1, cv2.LINE_AA)
            iou_text = ("frame IoU {:.3f}".format(frame_ious[index])
                        if frame_ious[index] is not None else "frame IoU - (no GT)")
            cv2.putText(footer, iou_text,
                        (8, 38), font, 0.43, viz.TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(footer, header, (width + 8, 18), font, 0.43,
                        viz.TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(footer, "NUDT-MIRSDT on Keii seq {}".format(metrics["sequence"]),
                        (width * 2 + 8, 18), font, 0.43, viz.TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(footer, "GT {}/{} frames".format(metrics["gt_frames"],
                                                         metrics["frames"]),
                        (width * 2 + 8, 38), font, 0.43, viz.TEXT_COLOR, 1, cv2.LINE_AA)
            row = np.vstack((np.hstack((ground_truth_panel, prediction_panel,
                                        combined_panel)), footer))
            writer.write(row)
            if index == 0:
                cv2.imwrite(str(preview_path), row)
    finally:
        writer.release()
    print("video: {}".format(output_path))
    print("preview: {}".format(preview_path))


if __name__ == "__main__":
    main()
