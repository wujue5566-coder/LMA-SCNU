"""Run LMAFormer inference on unlabeled infrared videos such as Datasets/Keii.

The training/evaluation pipeline needs ground-truth masks to build its dataset,
so the Keii clips (plain ``*.mp4`` files, no annotations) cannot go through
``inference_swin.py``.  This script decodes the videos, builds the same
5-frame sliding windows the model was trained with, runs the checkpoint, and
writes per-frame predictions plus overlay videos and statistics.

Because Keii has no ground truth, IoU / Pd / Fa cannot be computed.  The
reported numbers are detection proxies: predicted target count per frame,
detection persistence over time, peak probability, and mask stability.

Example::

    python inference_keii.py --model_path ./result/IRDST/checkpoint_IRDST.pth \
        --video_dir ./Datasets/Keii --output_dir ./predict/Keii --flip
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

from MFIRSTD.datasets import transforms as T
from MFIRSTD.utils.misc import NestedTensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Overlay colors (BGR), matching the palette used by visualize_irdst.py.
PREDICTION_COLOR = (70, 90, 245)
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (24, 28, 36)


def parse_args():
    from inference_swin import get_args_parser

    parser = argparse.ArgumentParser(
        "LMAFormer inference on unlabeled videos",
        parents=[get_args_parser()],
    )
    parser.add_argument("--video_dir", default="./Datasets/Keii",
                        help="Directory with input videos (*.mp4 / *.avi / *.mov).")
    parser.add_argument("--frames_dir", default=None,
                        help="Directory of image frames, or of one-sequence-per-subfolder "
                             "image folders. Overrides --video_dir.")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Only run these sequences, e.g. --sequences 1 3. "
                             "Default: every video in --video_dir.")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Limit frames per sequence (0 = all frames). Useful for a quick check.")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Predict every N-th frame (1 = every frame).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for the binary mask.")
    parser.add_argument("--min_area", type=int, default=2,
                        help="Minimum connected-component area (pixels) counted as a detection.")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip overlay video writing (useful for a fast benchmark).")
    parser.add_argument("--no_masks", action="store_true",
                        help="Skip saving per-frame probability / binary PNGs.")
    args = parser.parse_args()
    return args


def list_videos(video_dir: Path, sequences):
    extensions = (".mp4", ".avi", ".mov", ".mkv")
    videos = sorted(
        (path for path in video_dir.iterdir() if path.suffix.lower() in extensions),
        key=lambda path: (len(path.stem), path.stem),
    )
    if sequences:
        wanted = {str(name) for name in sequences}
        videos = [path for path in videos if path.stem in wanted]
    if not videos:
        raise FileNotFoundError("No videos found in {}".format(video_dir))
    return videos


def list_frame_sequences(frames_dir: Path, sequences):
    """Return image folders: the directory itself, or its per-sequence subfolders."""
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if any(path.suffix.lower() in image_extensions for path in frames_dir.iterdir() if path.is_file()):
        found = [frames_dir]
    else:
        found = sorted((path for path in frames_dir.iterdir()
                        if path.is_dir() and any(child.suffix.lower() in image_extensions
                                                 for child in path.iterdir())),
                       key=lambda path: (len(path.name), path.name))
    if sequences:
        wanted = {str(name) for name in sequences}
        found = [path for path in found if path.name in wanted]
    if not found:
        raise FileNotFoundError("No image sequences found in {}".format(frames_dir))
    return found


def read_video(path: Path):
    """Decode a video (or an image folder) into RGB uint8 frames plus fps."""
    if path.is_dir():
        files = sorted(name for name in path.iterdir()
                       if name.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        frames = [cv2.cvtColor(cv2.imread(str(name)), cv2.COLOR_BGR2RGB) for name in files]
        if not frames:
            raise RuntimeError("No image frames in {}".format(path))
        return frames, 10.0
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open video: {}".format(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError("No frames decoded from {}".format(path))
    return frames, float(fps)


def transform(clip, val_size, max_sc):
    """Replicate ValDataset's validation transforms for a list of RGB frames."""
    pil_images = [Image.fromarray(frame) for frame in clip]
    pil_images, _ = T.resize(pil_images, None, val_size,
                             max_size=int(max_sc * val_size))
    tensors = [F.normalize(F.to_tensor(image), mean=MEAN, std=STD) for image in pil_images]
    return torch.stack(tensors, dim=0)


def build_model(args, device):
    from MFIRSTD.models.lmaf_swin import build_model_lmaf_swinbackbone as build_model

    args.aux_loss = 0
    args.aux_loss_norm = 0
    model, _ = build_model(args)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _forward_scales(model, tensors, device, gt_shape, scales):
    """Average the sigmoid probability over the requested input scales."""
    probabilities = []
    for scale in scales:
        if scale == 1:
            scaled = tensors
        else:
            target = [int(round(value * scale)) for value in tensors.shape[-2:]]
            scaled = TF.interpolate(tensors, size=target, mode="bilinear", align_corners=False)
        mask = torch.zeros(scaled.shape[0], *scaled.shape[-2:], dtype=torch.bool, device=device)
        logits = model(NestedTensor(scaled.to(device), mask))["pred_masks"]
        probabilities.append(
            TF.interpolate(logits, size=gt_shape, mode="bilinear", align_corners=False).sigmoid())
    return torch.stack(probabilities, dim=0).mean(dim=0)


@torch.no_grad()
def predict_center(model, tensors, device, gt_shape, flip, threshold, scales=None):
    """Return the center-frame probability map as an HxW float32 array."""
    if scales is None:
        scales = [1]
    logits = _forward_scales(model, tensors, device, gt_shape, scales)
    if flip:
        logits_flipped = _forward_scales(model, tensors.flip(-1), device, gt_shape, scales)
        logits = (logits + logits_flipped.flip(-1)) / 2.0
    center_index = tensors.shape[0] // 2
    return logits[0, center_index].float().cpu().numpy()


def component_stats(mask, min_area):
    """Return (count, largest_area, total_area, largest centroid) for a mask."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
             if int(stats[i, cv2.CC_STAT_AREA]) >= min_area]
    if not areas:
        return 0, 0, 0, None
    total_area = sum(area for area, _ in areas)
    largest_area, largest_id = max(areas)
    return len(areas), largest_area, total_area, centroids[largest_id]


def draw_overlay(frame, probability, binary, min_area, frame_index, frame_total,
                 fps, max_prob, detections, threshold):
    """Build one visualization frame: mask overlay plus a metric footer."""
    view = frame.copy()
    highlight = binary.astype(bool)
    if highlight.any():
        view[highlight] = ((1.0 - 0.45) * view[highlight]
                           + 0.45 * np.asarray(PREDICTION_COLOR)).astype(np.uint8)
    count, _, _, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    for component in range(1, count):
        x, y, width, height, area = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8)[2][component]
        if int(area) < min_area:
            continue
        size = max(width, height, 10)
        x1 = max(0, x - (size - width) // 2)
        y1 = max(0, y - (size - height) // 2)
        x2 = min(view.shape[1] - 1, x1 + size - 1)
        y2 = min(view.shape[0] - 1, y1 + size - 1)
        cv2.rectangle(view, (x1, y1), (x2, y2), PREDICTION_COLOR, 1, lineType=cv2.LINE_AA)

    footer_height = 40
    footer = np.full((footer_height, view.shape[1], 3), PANEL_COLOR, dtype=np.uint8)
    cv2.putText(footer, "Frame {}/{}  px>{}".format(frame_index, frame_total, threshold),
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(footer, "peak {:.3f}  detections {}".format(max_prob, detections),
                (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)
    return np.vstack((view, footer))


def frame_iou(mask_a, mask_b):
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def process_sequence(model, video_path, args, device, out_root):
    frames, fps = read_video(video_path)
    total_frames = len(frames) if args.max_frames <= 0 else min(len(frames), args.max_frames)
    indices = list(range(0, total_frames, args.frame_stride))
    height, width = frames[0].shape[:2]
    half = args.num_frames // 2
    seq_dir = out_root / video_path.stem
    logits_dir = seq_dir / "logits"
    masks_dir = seq_dir / "bin_mask"
    seq_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_masks:
        logits_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(seq_dir / "overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height + 40),
        )
        if not writer.isOpened():
            raise RuntimeError("Unable to open video writer for {}".format(seq_dir))

    rows = []
    previous_mask = None
    previous_centroid = None
    scales = [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3] if args.msc else [1]
    start = time.time()
    for order, center in enumerate(tqdm(indices, desc=video_path.stem, unit="frame", leave=False)):
        clip = [frames[(center + offset - half) % total_frames] for offset in range(args.num_frames)]
        tensors = transform(clip, args.val_size, args.max_sc)
        probability = predict_center(model, tensors, device, (height, width),
                                     args.flip, args.threshold, scales)
        binary = (probability > args.threshold).astype(np.uint8)
        detections, largest_area, total_area, centroid = component_stats(binary, args.min_area)
        stability = None if previous_mask is None else frame_iou(binary, previous_mask)
        drift = None
        if centroid is not None and previous_centroid is not None:
            drift = float(np.linalg.norm(centroid - previous_centroid))
        previous_mask = binary
        previous_centroid = centroid
        rows.append({
            "frame_index": center,
            "max_prob": float(probability.max()),
            "mean_prob": float(probability.mean()),
            "detections": detections,
            "largest_area": largest_area,
            "total_area": total_area,
            "centroid_x": None if centroid is None else float(centroid[0]),
            "centroid_y": None if centroid is None else float(centroid[1]),
            "temporal_iou": stability,
            "centroid_drift": drift,
        })
        if not args.no_masks:
            cv2.imwrite(str(logits_dir / "f{:06d}.png".format(center)),
                        (probability * 255.0).clip(0, 255).astype(np.uint8))
            cv2.imwrite(str(masks_dir / "f{:06d}.png".format(center)),
                        (binary * 255).astype(np.uint8))
        if writer is not None:
            overlay = draw_overlay(frames[center], probability, binary, args.min_area,
                                   center, total_frames, fps, float(probability.max()),
                                   detections, args.threshold)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.release()

    elapsed = time.time() - start
    with open(seq_dir / "frames.csv", "w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    detection_flags = [row["detections"] > 0 for row in rows]
    stabilities = [row["temporal_iou"] for row in rows if row["temporal_iou"] is not None]
    drifts = [row["centroid_drift"] for row in rows if row["centroid_drift"] is not None]
    summary = {
        "video": video_path.stem,
        "frames_decoded": len(frames),
        "frames_evaluated": len(rows),
        "height": height,
        "width": width,
        "fps": fps,
        "detection_rate": float(np.mean(detection_flags)),
        "mean_detections": float(np.mean([row["detections"] for row in rows])),
        "mean_largest_area": float(np.mean([row["largest_area"] for row in rows])),
        "mean_max_prob": float(np.mean([row["max_prob"] for row in rows])),
        "frames_prob_above_0.9": int(np.sum([row["max_prob"] > 0.9 for row in rows])),
        "mean_temporal_iou": float(np.mean(stabilities)) if stabilities else None,
        "mean_centroid_drift": float(np.mean(drifts)) if drifts else None,
        "seconds": elapsed,
    }
    logger.debug("sequence %s done in %.1fs", video_path.stem, elapsed)
    return summary


def main():
    args = parse_args()
    if not hasattr(args, "max_sc") or args.max_sc is None:
        args.max_sc = 1.8
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    video_dir = Path(args.video_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    handling = logging.FileHandler(out_root / "out.log", encoding="utf-8")
    handling.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s"))
    logging.getLogger().addHandler(handling)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.DEBUG)

    videos = (list_frame_sequences(Path(args.frames_dir), args.sequences)
              if args.frames_dir else list_videos(video_dir, args.sequences))
    logger.debug("checkpoint: %s", args.model_path)
    logger.debug("videos: %s", ", ".join(path.stem for path in videos))
    logger.debug("device: %s | val_size=%d | flip=%s | msc=%s",
                 device, args.val_size, args.flip, args.msc)
    model = build_model(args, device)

    summaries = []
    for video in videos:
        summaries.append(process_sequence(model, video, args, device, out_root))

    with open(out_root / "summary.json", "w") as handle:
        json.dump(summaries, handle, indent=2)
    with open(out_root / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    total_frames = sum(item["frames_evaluated"] for item in summaries)
    total_seconds = sum(item["seconds"] for item in summaries)
    logger.debug("all sequences: %d frames in %.1fs (%.2f s/frame)",
                 total_frames, total_seconds, total_seconds / max(total_frames, 1))
    for item in summaries:
        logger.debug("%s: frames=%d detection_rate=%.3f mean_detections=%.2f "
                     "mean_max_prob=%.3f mean_temporal_iou=%s",
                     item["video"], item["frames_evaluated"], item["detection_rate"],
                     item["mean_detections"], item["mean_max_prob"],
                     "n/a" if item["mean_temporal_iou"] is None
                     else "%.3f" % item["mean_temporal_iou"])
    print("\nSummary written to {}".format(out_root / "summary.csv"))


if __name__ == "__main__":
    main()
