"""Sample a Keii clip at a fixed time interval and keep N frames.

Splits the clip into ``count`` equally spaced positions, writes those frames as
PNG (named after the source frame index) plus a ``sampling.csv`` log, and renders
a plain slide-show video **without** any burned-in footer, so the result can be
fed straight back into ``inference_keii.py``.

Example::

    python make_uniform_frames.py --video ./Datasets/Keii/7.mp4 \
        --output_dir ./Datasets/Keii/7/uniform100 --count 100 --fps 10
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Uniform time sampling of a Keii clip")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Folder for the frames, the log and the clean video.")
    parser.add_argument("--count", type=int, default=100,
                        help="How many frames to keep (evenly spaced in time).")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frame rate of the slide-show video.")
    parser.add_argument("--video_name", default=None,
                        help="Output video name (default: uniform_<count>.mp4).")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError("unable to open {}".format(args.video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("clip reports no frames")
    if args.count > total:
        raise ValueError("--count {} exceeds the {} frames in the clip".format(
            args.count, total))

    wanted = sorted({int(round(value))
                     for value in np.linspace(0, total - 1, args.count)})
    if len(wanted) != args.count:
        raise RuntimeError("could not build {} distinct indices".format(args.count))

    video_path = output_dir / (args.video_name or "uniform_{}.mp4".format(args.count))
    writer = None
    rows = []
    position = 0
    pending = set(wanted)
    while pending:
        ok, frame = capture.read()
        if not ok:
            break
        if position in pending:
            pending.discard(position)
            name = "{:06d}.png".format(position)
            cv2.imwrite(str(output_dir / name), frame)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(video_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                         (width, height))
                if not writer.isOpened():
                    raise RuntimeError("unable to open writer for {}".format(video_path))
            writer.write(frame)
            rows.append((len(rows), position, name))
        position += 1
    capture.release()
    if writer is not None:
        writer.release()
    if pending:
        raise RuntimeError("clip ended before frames {}".format(sorted(pending)[:5]))

    log_path = output_dir / "sampling.csv"
    with open(log_path, "w", newline="") as handle:
        writer_csv = csv.writer(handle)
        writer_csv.writerow(["order", "source_frame", "file"])
        writer_csv.writerows(rows)

    gaps = np.diff([row[1] for row in rows])
    print("source clip    : {} ({} frames)".format(args.video, total))
    print("frames kept    : {}".format(len(rows)))
    print("sampling step  : mean {:.2f} median {} min {} max {}".format(
        gaps.mean(), int(np.median(gaps)), gaps.min(), gaps.max()))
    print("first / last   : {} / {}".format(rows[0][1], rows[-1][1]))
    print("frames folder  : {}".format(output_dir))
    print("log            : {}".format(log_path))
    print("clean video    : {}".format(video_path))


if __name__ == "__main__":
    main()
