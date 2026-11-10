"""Create side-by-side IRDST prediction visualization videos.

Each output video contains three synchronized panels for every frame:
  * left: the ground-truth mask overlaid on the infrared image;
  * middle: the predicted mask overlaid on the infrared image;
  * right: both masks overlaid on the infrared image;
  * bottom-left: the live frame counter; bottom-right: compact Pd/Fa metrics;
  * bounding boxes around ground-truth and predicted connected components.

Run from the repository root with the IRSTD environment, for example::

    python visualize_irdst.py --video 13 --max_frames 10
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from MFIRSTD.datasets import path_config


# BGR colors used by OpenCV.  Ground truth is yellow, missed targets are
# blue, prediction boxes are red, and false alarms are magenta.
GROUND_TRUTH_COLOR = (0, 220, 220)     # GT: yellow
PREDICTION_COLOR = (70, 90, 245)       # prediction/boxes: red
TRUE_POSITIVE_COLOR = (60, 220, 100)   # TP: correctly detected target (green)
MISSED_TARGET_COLOR = (255, 0, 0)      # FN: ground truth without prediction (blue)
FALSE_ALARM_COLOR = (255, 0, 180)      # FP: prediction outside ground truth (magenta)
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (24, 28, 36)


def natural_key(path):
    """Sort names such as ``1(2).png`` before ``1(10).png``."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("Unable to read mask: {}".format(path))
    return mask > 0


def iou_score(annotation, segmentation):
    annotation = np.asarray(annotation, dtype=bool)
    segmentation = np.asarray(segmentation, dtype=bool)
    union = np.logical_or(annotation, segmentation).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(annotation, segmentation).sum() / union)


def connected_boxes(mask, min_area):
    """Return external-component boxes and component count after filtering."""
    binary = np.asarray(mask, dtype=np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if int(area) >= min_area:
            boxes.append((int(x), int(y), int(width), int(height), int(area)))
    return boxes


def overlay_mask(image, mask, color, alpha=0.38):
    result = image.copy()
    result[mask] = ((1.0 - alpha) * result[mask] + alpha * np.asarray(color)).astype(np.uint8)
    return result


def overlay_masks(image, ground_truth, prediction, alpha=0.38):
    """Show TP/FN/FP with distinct colors in the overlay panel.

    TP (overlap) is green, FN (ground truth only; missed target) is blue,
    and FP (prediction only; false alarm) is magenta.
    """
    result = image.copy()
    regions = (
        (ground_truth & prediction, TRUE_POSITIVE_COLOR),
        (ground_truth & ~prediction, MISSED_TARGET_COLOR),
        (~ground_truth & prediction, FALSE_ALARM_COLOR),
    )
    for region, color in regions:
        result[region] = ((1.0 - alpha) * result[region] + alpha * np.asarray(color)).astype(np.uint8)
    return result


def split_prediction_boxes(ground_truth, prediction, min_area, match_distance=3.0):
    """Split predicted boxes into matched P boxes and false-alarm F boxes."""
    gt_count, _, _, gt_centroids = cv2.connectedComponentsWithStats(
        ground_truth.astype(np.uint8), connectivity=8)
    pred_count, _, pred_stats, pred_centroids = cv2.connectedComponentsWithStats(
        prediction.astype(np.uint8), connectivity=8)
    available = set(range(1, pred_count))
    matched = set()
    for gt_id in range(1, gt_count):
        if not available:
            break
        pred_id = min(
            available,
            key=lambda candidate: np.linalg.norm(
                gt_centroids[gt_id] - pred_centroids[candidate]))
        if np.linalg.norm(gt_centroids[gt_id] - pred_centroids[pred_id]) < match_distance:
            matched.add(pred_id)
            available.remove(pred_id)

    matched_boxes = []
    false_alarm_boxes = []
    for pred_id in range(1, pred_count):
        x, y, width, height, area = pred_stats[pred_id]
        if int(area) < min_area:
            continue
        box = (int(x), int(y), int(width), int(height), int(area))
        if pred_id in matched:
            matched_boxes.append(box)
        else:
            false_alarm_boxes.append(box)
    return matched_boxes, false_alarm_boxes


def draw_overlay_legend(image):
    """Add a compact legend for TP, missed targets (FN), and false alarms (FP)."""
    cv2.rectangle(image, (0, 0), (image.shape[1], 40), PANEL_COLOR, -1)
    cv2.putText(image, "Overlay", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                TEXT_COLOR, 1, cv2.LINE_AA)
    entries = (("TP", TRUE_POSITIVE_COLOR), ("FN miss", MISSED_TARGET_COLOR),
               ("FP false", FALSE_ALARM_COLOR))
    x = 62
    for label, color in entries:
        cv2.rectangle(image, (x, 8), (x + 7, 16), color, -1)
        cv2.putText(image, label, (x + 10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    TEXT_COLOR, 1, cv2.LINE_AA)
        x += 42


def draw_boxes(image, boxes, color, prefix):
    """Draw component boxes, enlarging only tiny boxes for visibility."""
    image_height, image_width = image.shape[:2]
    min_display_size = 10
    for index, (x, y, width, height, area) in enumerate(boxes, start=1):
        display_width = max(width, min_display_size)
        display_height = max(height, min_display_size)
        x1 = max(0, x - (display_width - width) // 2)
        y1 = max(0, y - (display_height - height) // 2)
        x2 = min(image_width - 1, x1 + display_width - 1)
        y2 = min(image_height - 1, y1 + display_height - 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
        label = "{}{}".format(prefix, index)
        text_y = max(12, y1 - 3)
        cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    color, 1, cv2.LINE_AA)


def fit_image(image, size):
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def put_label(image, text, origin, color):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                TEXT_COLOR, 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                color, 1, cv2.LINE_AA)


def load_video_ious(csv_path):
    """Read the existing per-video IoU CSV, if it is available."""
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", newline="") as file:
        rows = list(csv.reader(file))
    if len(rows) < 2:
        return {}
    result = {}
    for name, value in zip(rows[0], rows[1]):
        try:
            result[name.strip()] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def count_detected_targets(ground_truth, prediction, match_distance=3.0):
    """Count GT targets detected by one-to-one centroid matching.

    The 3-pixel matching radius follows the project's PD definition.  A
    prediction can match at most one ground-truth target.
    """
    gt_count, _, _, gt_centroids = cv2.connectedComponentsWithStats(
        ground_truth.astype(np.uint8), connectivity=8)
    pred_count, _, _, pred_centroids = cv2.connectedComponentsWithStats(
        prediction.astype(np.uint8), connectivity=8)
    total_targets = gt_count - 1
    available_predictions = set(range(1, pred_count))
    detected_targets = 0
    for gt_id in range(1, gt_count):
        if not available_predictions:
            break
        candidates = min(
            available_predictions,
            key=lambda pred_id: np.linalg.norm(gt_centroids[gt_id] - pred_centroids[pred_id]),
        )
        distance = np.linalg.norm(gt_centroids[gt_id] - pred_centroids[candidates])
        if distance < match_distance:
            detected_targets += 1
            available_predictions.remove(candidates)
    return detected_targets, total_targets


def accumulate_pd_fa(image_paths, mask_dir, prediction_dir):
    """Accumulate Pd/Fa numerators and denominators.

    Pd = detected true targets / total true targets.
    Fa = false detected non-target pixels / total image pixels.
    """
    detected_targets = 0
    total_targets = 0
    false_pixels = 0
    total_pixels = 0
    for image_path in image_paths:
        name = image_path.name
        ground_truth = read_mask(mask_dir / name)
        prediction = read_mask(prediction_dir / name)
        detected, targets = count_detected_targets(ground_truth, prediction)
        detected_targets += detected
        total_targets += targets
        false_pixels += int(np.logical_and(prediction, ~ground_truth).sum())
        total_pixels += int(prediction.size)
    return detected_targets, total_targets, false_pixels, total_pixels


def calculate_pd_fa(detected_targets, total_targets, false_pixels, total_pixels):
    pd_value = detected_targets / total_targets if total_targets else None
    fa_value = false_pixels / total_pixels if total_pixels else None
    return pd_value, fa_value


def compute_video_pd_fa(image_paths, mask_dir, prediction_dir):
    return calculate_pd_fa(*accumulate_pd_fa(image_paths, mask_dir, prediction_dir))


def compute_dataset_pd_fa(video_names, image_root, mask_root, prediction_root):
    sums = np.zeros(4, dtype=np.float64)
    for video_name in video_names:
        image_paths = sorted((image_root / video_name).glob("*.png"), key=natural_key)
        sums += np.asarray(accumulate_pd_fa(
            image_paths, mask_root / video_name, prediction_root / video_name), dtype=np.float64)
    return calculate_pd_fa(*sums)


def read_video_names(video_file, selected_video):
    if selected_video:
        return [str(selected_video)]
    with video_file.open("r") as file:
        return [line.strip() for line in file if line.strip()]


def make_video(video_name, image_root, mask_root, prediction_root, output_root,
               video_iou, video_pd, video_fa, dataset_pd, dataset_fa,
               fps, max_frames, min_area):
    image_dir = image_root / video_name
    mask_dir = mask_root / video_name
    prediction_dir = prediction_root / video_name
    image_paths = sorted(image_dir.glob("*.png"), key=natural_key)
    if not image_paths:
        raise FileNotFoundError("No input frames found in {}".format(image_dir))

    frame_names = {path.name for path in image_paths}
    missing = [name for name in frame_names
               if not (mask_dir / name).is_file() or not (prediction_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing mask/prediction for video {}: {}".format(video_name, missing[:3]))
    if max_frames > 0:
        image_paths = image_paths[:max_frames]

    first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError("Unable to read image: {}".format(image_paths[0]))
    height, width = first_image.shape[:2]
    panel_width = width
    # Reserve a compact two-line strip for per-video and whole-dataset metrics.
    footer_height = 48
    output_size = (panel_width * 3, height + footer_height)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "video_{}.mp4".format(video_name)
    preview_path = output_root / "video_{}_first_frame.png".format(video_name)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, output_size)
    if not writer.isOpened():
        raise RuntimeError("Unable to open video writer for {}".format(output_path))

    frame_ious = []
    try:
        for image_path in image_paths:
            name = image_path.name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            ground_truth = read_mask(mask_dir / name)
            prediction = read_mask(prediction_dir / name)
            if image is None:
                raise FileNotFoundError("Unable to read image: {}".format(image_path))
            if ground_truth.shape != image.shape[:2]:
                ground_truth = cv2.resize(ground_truth.astype(np.uint8),
                                          (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            if prediction.shape != image.shape[:2]:
                prediction = cv2.resize(prediction.astype(np.uint8),
                                        (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

            frame_iou = iou_score(ground_truth, prediction)
            frame_ious.append(frame_iou)
            ground_truth_boxes = connected_boxes(ground_truth, min_area)
            prediction_boxes = connected_boxes(prediction, min_area)
            matched_prediction_boxes, false_alarm_boxes = split_prediction_boxes(
                ground_truth, prediction, min_area)

            ground_truth_panel = overlay_mask(image, ground_truth, GROUND_TRUTH_COLOR)
            prediction_panel = overlay_mask(image, prediction, PREDICTION_COLOR)
            combined_panel = overlay_masks(image, ground_truth, prediction)
            draw_boxes(ground_truth_panel, ground_truth_boxes, GROUND_TRUTH_COLOR, "GT-")
            draw_boxes(prediction_panel, prediction_boxes, PREDICTION_COLOR, "P-")
            draw_boxes(combined_panel, ground_truth_boxes, MISSED_TARGET_COLOR, "GT-")
            draw_boxes(combined_panel, matched_prediction_boxes, PREDICTION_COLOR, "P-")
            draw_boxes(combined_panel, false_alarm_boxes, FALSE_ALARM_COLOR, "F-")
            put_label(ground_truth_panel, "Ground truth", (10, 27), GROUND_TRUTH_COLOR)
            put_label(prediction_panel, "Prediction", (10, 27), PREDICTION_COLOR)
            draw_overlay_legend(combined_panel)

            body = np.hstack((ground_truth_panel, prediction_panel, combined_panel))
            footer = np.full((footer_height, output_size[0], 3), PANEL_COLOR, dtype=np.uint8)
            frame_text = "Frame {}/{}".format(len(frame_ious), len(image_paths))
            def format_metrics(pd_metric, fa_metric):
                fa_text = ("{:.1e}".format(fa_metric).replace("e-0", "e-").replace("e+0", "e+")
                           if fa_metric is not None else "N/A")
                return "Pd {:.3f} | Fa {}".format(
                    pd_metric if pd_metric is not None else float("nan"), fa_text)
            video_metric_text = "Video: " + format_metrics(video_pd, video_fa)
            dataset_metric_text = "Dataset: " + format_metrics(dataset_pd, dataset_fa)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.43
            thickness = 1
            cv2.putText(footer, frame_text, (8, 18), font, font_scale,
                        TEXT_COLOR, thickness, cv2.LINE_AA)
            cv2.putText(footer, video_metric_text, (panel_width + 8, 18), font,
                        font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
            cv2.putText(footer, dataset_metric_text, (panel_width * 2 + 8, 18), font,
                        font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
            frame = np.vstack((body, footer))
            writer.write(frame)
            if len(frame_ious) == 1:
                cv2.imwrite(str(preview_path), frame)
    finally:
        writer.release()

    return output_path, preview_path, len(frame_ious), float(np.mean(frame_ious))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize IRDST masks and target boxes.")
    parser.add_argument("--prediction_dir", default="./predict/IRDST",
                        help="Inference output directory containing IRDST/bin_mask.")
    parser.add_argument("--output_dir", default="./predict/IRDST/visualizations",
                        help="Directory for generated MP4 files and PNG previews.")
    parser.add_argument("--video", default=None,
                        help="Only process one video, for example 13; default is all validation videos.")
    parser.add_argument("--fps", type=float, default=10,
                        help="Output video frame rate (default: 10).")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Maximum frames per video; 0 means all frames.")
    parser.add_argument("--min_area", type=int, default=1,
                        help="Minimum connected-component area for boxes (default: 1 pixel).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_frames < 0:
        raise ValueError("--max_frames cannot be negative")
    if args.min_area < 1:
        raise ValueError("--min_area must be at least 1")

    prediction_dir = Path(args.prediction_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_root = Path(path_config.img_path)
    mask_root = Path(path_config.mask_path)
    video_file = Path(path_config.val_seqs_file)
    prediction_root = prediction_dir / "IRDST" / "bin_mask"
    csv_path = prediction_dir / "dataset_results.csv"
    if not csv_path.is_file():
        csv_path = prediction_dir / "IRDST" / "dataset_results.csv"

    for path in (image_root, mask_root, video_file, prediction_root):
        if not path.exists():
            raise FileNotFoundError("Required path does not exist: {}".format(path))

    video_ious = load_video_ious(csv_path)
    all_video_names = read_video_names(video_file, None)
    video_names = read_video_names(video_file, args.video)
    # Keep Dataset Pd/Fa as the full validation-set value even when --video
    # is used to generate only one preview video.
    dataset_pd, dataset_fa = compute_dataset_pd_fa(
        all_video_names, image_root, mask_root, prediction_root)
    print("Dataset root: {}".format(image_root.parent))
    print("Videos to process: {}".format(", ".join(video_names)))
    print("Prediction masks: {}".format(prediction_root))

    generated = []
    for video_name in video_names:
        image_paths = sorted((image_root / video_name).glob("*.png"), key=natural_key)
        video_pd, video_fa = compute_video_pd_fa(
            image_paths, mask_root / video_name, prediction_root / video_name)
        result = make_video(video_name, image_root, mask_root, prediction_root, output_dir,
                            video_ious.get(video_name), video_pd, video_fa,
                            dataset_pd, dataset_fa, args.fps, args.max_frames, args.min_area)
        generated.append(result)
        output_path, preview_path, frame_count, calculated_iou = result
        print("{}: {} frames, IoU {:.5f}, Pd {:.4f}, Fa {:.2e}".format(
            video_name, frame_count, calculated_iou,
            video_pd if video_pd is not None else 0.0,
            video_fa if video_fa is not None else 0.0))
        print("  video: {}".format(output_path))
        print("  preview: {}".format(preview_path))
    print("Generated {} video(s) in {}".format(len(generated), output_dir))


if __name__ == "__main__":
    main()
