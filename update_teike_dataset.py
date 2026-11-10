"""Incrementally maintain the TeiKe dataset and only test the new sequences.

The dataset lives outside the repo (``$LMAFORMER_TEIKE_ROOT``) with this
convention, one folder per sequence::

    data/<seq>/
        images/000000.png ... 000099.png     # sampled frames, consecutive names
        images/000000.json ...               # Labelme annotations (same stem)
        masks/000000.png  ...                # ground truth, only for annotated frames
        uniform_100.mp4                      # clean slideshow of the kept frames
        sampling.csv                         # order -> source frame -> file name

Newly shot clips land in ``demo/<n>.mp4``.  This script works on the *new*
sequence numbers only, so an existing 1..48 never has to be redone:

    --scan      (default) report what exists, what is missing, what is pending
    --add       sample the pending raw clips into data/<n>/ and re-run the
                rename/archive step (calls make_uniform_frames.py + uniform_frames.py)
    --mask      export masks incrementally for sequences that have JSON annotations
                (calls json2mask.py; existing masks are skipped)
    --detect    run detection only for sequences without an overlay video yet
                (calls inference_keii.py with just those sequence names)
    --eval      quantitative test (Pd/Fa/AUC + GT comparison video) on sequences that
                have masks but no metrics yet (calls eval_keii_masked.py and then
                report_teike_eval.py to pool everything)
    --all       run add -> mask -> detect -> eval in one go for everything pending
    --lmformer  refresh the LMAFormer-format root (junctions) and append the new
                sequence numbers to ImageSets/val_new.txt (train list untouched)

Examples::

    python update_teike_dataset.py --scan
    python update_teike_dataset.py --add
    python update_teike_dataset.py --mask
    python update_teike_dataset.py --detect
    python update_teike_dataset.py --eval
    python update_teike_dataset.py --all

Add ``--dry_run`` to print the commands without executing them.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from project_paths import TEIKE_ROOT, create_directory_link

REPO = Path(__file__).resolve().parent
MAKE_UNIFORM = REPO / "make_uniform_frames.py"


def parse_args():
    parser = argparse.ArgumentParser("Incremental update of the TeiKe dataset")
    parser.add_argument("--teike_root", type=Path, default=TEIKE_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--count", type=int, default=100, help="frames per sequence")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--detect_out", default=str(REPO / "predict" / "TeiKe_NUDT_detection"))
    parser.add_argument("--eval_out", default=str(REPO / "predict" / "TeiKe_NUDT"),
                        help="Where --eval writes metrics and GT comparison videos "
                             "(existing per-sequence results are skipped).")
    parser.add_argument("--eval_seqs", nargs="*", default=None,
                        help="Only evaluate these sequences (default: sequences that have "
                             "masks but no metrics yet).")
    parser.add_argument("--official", action="store_true",
                        help="With --eval: also run the repo's inference_swin.py on the same "
                             "sequences for a cross-check.")
    parser.add_argument("--official_out", default=str(REPO / "predict" / "TeiKe_official"))
    parser.add_argument("--staging", default=str(REPO / "predict" / "TeiKe_detect_input"),
                        help="Folder of <seq>.mp4 clips fed to inference_keii.py")
    parser.add_argument("--model_path", default="./result/NUDT-MIRSDT/checkpoint_NUDT-MIRSDT.pth")
    parser.add_argument("--keep_masks", action="store_true",
                        help="With --detect/--all: also write the per-frame prediction "
                             "masks (<out>/<seq>/logits, bin_mask). Off by default, which "
                             "keeps only the overlay video and the per-frame CSV.")
    parser.add_argument("--lmformer_dst", default=str(REPO / "Datasets" / "TeiKe_LMAFormer"))
    parser.add_argument("--force", action="store_true",
                        help="Redo sequences that already look processed.")
    parser.add_argument("--dry_run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scan", action="store_true", help="report only (default)")
    mode.add_argument("--add", action="store_true")
    mode.add_argument("--mask", action="store_true")
    mode.add_argument("--detect", action="store_true")
    mode.add_argument("--eval", action="store_true")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--lmformer", action="store_true")
    return parser.parse_args()


def run(command, dry_run):
    printable = " ".join('"{}"'.format(part) if " " in str(part) else str(part)
                         for part in command)
    if dry_run:
        print("  [dry-run] {}".format(printable))
        return
    print("  $ {}".format(printable), flush=True)
    result = subprocess.run([str(part) for part in command], cwd=str(REPO))
    if result.returncode != 0:
        raise SystemExit("command failed with code {}".format(result.returncode))


def sequence_dirs(teike_root):
    data_root = teike_root / "data"
    if not data_root.is_dir():
        raise SystemExit("no data/ under {}".format(teike_root))
    return data_root, sorted((path for path in data_root.iterdir()
                              if path.is_dir() and path.name.isdigit()),
                             key=lambda path: int(path.name))


def survey(teike_root, detect_out, force):
    data_root, sequences = sequence_dirs(teike_root)
    rows = []
    for path in sequences:
        images = len(list((path / "images").glob("*.png")))
        annotations = len(list((path / "images").glob("*.json")))
        masks = len(list((path / "masks").glob("*.png"))) if (path / "masks").is_dir() else 0
        sampled = (path / "uniform_100.mp4").is_file()
        detected = (Path(detect_out) / path.name / "overlay.mp4").is_file()
        if masks and masks >= images:
            state = "已标注"
        elif masks:
            state = "掩码部分({}/{})".format(masks, images)
        elif annotations:
            state = "已标待导出({}/{})".format(annotations, images)
        else:
            state = "未标注"
        rows.append({"seq": path.name, "images": images, "json": annotations,
                     "masks": masks, "sampled": sampled, "detected": detected,
                     "state": state})

    demo_root = teike_root / "demo"
    pending = []
    if demo_root.is_dir():
        existing = {path.name for path in sequences}
        for video in sorted(demo_root.glob("*.mp4")):
            if not video.stem.isdigit():
                continue
            target = data_root / video.stem
            if video.stem in existing and (target / "images").is_dir() and not force:
                continue
            pending.append(video)
    return data_root, rows, pending


def print_survey(rows, pending):
    print("{:<5}{:>8}{:>7}{:>7}{:>9}{:>11}   {}".format(
        "seq", "images", "json", "masks", "抽帧", "检测视频", "状态"))
    for row in rows:
        print("{:<5}{:>8}{:>7}{:>7}{:>9}{:>11}   {}".format(
            row["seq"], row["images"], row["json"], row["masks"],
            "有" if row["sampled"] else "-", "有" if row["detected"] else "-",
            row["state"]))
    fully = [row["seq"] for row in rows if row["state"] == "已标注"]
    total_masks = sum(row["masks"] for row in rows)
    print("\n共 {} 段 | 已标注 {} 段 | 掩码合计 {} 张 | 检测视频 {} 段".format(
        len(rows), len(fully), total_masks,
        sum(1 for row in rows if row["detected"])))
    if pending:
        print("待抽帧的新视频: {}".format(", ".join(video.name for video in pending)))
    else:
        print("待抽帧的新视频: 无")


def add_pending(teike_root, pending, args):
    if not pending:
        print("没有需要抽帧的新视频。把新拍的视频放进 {} 再跑一次。".format(
            teike_root / "demo"))
        return
    for video in pending:
        target = teike_root / "data" / video.stem
        target.mkdir(parents=True, exist_ok=True)
        print("[抽帧] {} -> {}".format(video.name, target))
        run([args.python, MAKE_UNIFORM, "--video", video,
             "--output_dir", target, "--count", args.count, "--fps", args.fps],
            args.dry_run)
    print("\n[改名+存档] 调你自己的 uniform_frames.py（已改过名的段会被自动跳过）")
    run([args.python, teike_root / "uniform_frames.py",
         "--root", teike_root / "data", "--out", teike_root, "--execute"],
        args.dry_run)
    print("\n下一步：用 Labelme 标注 data/<新编号>/images/*.png（存成同名 .json），"
          "然后跑 --mask 导出掩码。")


def export_masks(teike_root, rows, args):
    todo = [row for row in rows
            if row["json"] > 0 and (args.force or row["masks"] < row["json"])]
    if not todo:
        print("没有需要导出掩码的段（要么还没标 JSON，要么已经导出过）。")
        return
    for row in todo:
        print("[导出掩码] seq {}  json={} masks={}".format(
            row["seq"], row["json"], row["masks"]))
        command = [args.python, teike_root / "json2mask.py",
                   "--root", teike_root / "data" / row["seq"]]
        if args.force:
            command.append("--overwrite")
        run(command, args.dry_run)
    print("\n提醒：json2mask.py 默认跳过已存在的掩码；要整段重导加 --force。")


def detect_new(teike_root, rows, args):
    todo = [row["seq"] for row in rows
            if (teike_root / "data" / row["seq"] / "uniform_100.mp4").is_file()
            and (args.force or not row["detected"])]
    if not todo:
        print("所有段都已经有检测视频了，无新增。")
        return
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    for seq in todo:
        source = teike_root / "data" / seq / "uniform_100.mp4"
        target = staging / "{}.mp4".format(seq)
        if args.force or not target.is_file():
            print("[整理输入] {}".format(target.name))
            if not args.dry_run:
                shutil.copy2(source, target)
    print("\n[检测] 只跑新增 {} 段: {}".format(len(todo), ", ".join(todo)))
    command = [args.python, REPO / "inference_keii.py",
               "--model_path", args.model_path,
               "--video_dir", staging,
               "--sequences", *todo,
               "--output_dir", args.detect_out,
               "--flip"]
    if args.keep_masks:
        print("  （同时写出逐帧预测掩码 logits/ 与 bin_mask/）")
    else:
        print("  （只出视频和 frames.csv；要预测掩码加 --keep_masks）")
        command.append("--no_masks")
    run(command, args.dry_run)


def evaluate(teike_root, rows, args):
    """Quantitative test (Pd/Fa/AUC + GT comparison video) for the new sequences."""
    eval_out = Path(args.eval_out)
    if args.eval_seqs:
        todo = list(args.eval_seqs)
    else:
        todo = [row["seq"] for row in rows
                if row["masks"] > 0
                and (args.force or not (eval_out / row["seq"] / "metrics.json").is_file())]
    if not todo:
        print("没有需要评测的段（要么还没导出掩码，要么都已经评测过）。")
        return
    print("[评测] 共 {} 段: {}".format(len(todo), ", ".join(todo)))
    evaluated = []
    for seq in todo:
        masks = teike_root / "data" / seq / "masks"
        if not masks.is_dir() or not any(masks.glob("*.png")):
            print("  [跳过] seq {} 还没有掩码".format(seq))
            continue
        print("[seq {}]".format(seq))
        run([args.python, REPO / "eval_keii_masked.py",
             "--sequence", seq,
             "--images_dir", teike_root / "data" / seq / "images",
             "--masks_dir", masks,
             "--model_path", args.model_path,
             "--out_root", eval_out,
             "--fps", args.fps], args.dry_run)
        evaluated.append(seq)

    if evaluated:
        print("\n[汇总] 合并 {} 下所有段为数据集级指标".format(eval_out))
        run([args.python, REPO / "report_teike_eval.py",
             "--run_root", eval_out,
             "--data_root", teike_root / "data",
             "--output", eval_out / "dataset_summary.png"], args.dry_run)
    if args.official and evaluated:
        cross_check(teike_root, evaluated, args)


def cross_check(teike_root, sequences, args):
    """Also run the repo's own inference_swin.py on the same sequences."""
    import os

    eval_root = Path(args.official_out) / "_eval_root"
    print("\n[官方交叉验证] 评测根 {}（val_new.txt 只列这几段）".format(eval_root))
    (eval_root / "ImageSets").mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        for kind in ("images", "masks"):
            link = eval_root / kind / seq
            target = teike_root / "data" / seq / kind
            if link.exists() or args.dry_run or not target.is_dir():
                continue
            create_directory_link(target, link)
    if not args.dry_run:
        (eval_root / "ImageSets" / "val_new.txt").write_text(
            "\n".join(sequences) + "\n", encoding="utf-8")
    command = [args.python, REPO / "inference_swin.py",
               "--model_path", args.model_path,
               "--dataset", "TeiKe",
               "--output_dir", Path(args.official_out) / "cross_check",
               "--flip", "--num_workers", "0"]
    printable = " ".join(str(part) for part in command)
    if args.dry_run:
        print("  [dry-run] LMAFORMER_DATASET_ROOT={} {}".format(eval_root, printable))
        return
    print("  $ LMAFORMER_DATASET_ROOT={} {}".format(eval_root, printable), flush=True)
    environment = dict(os.environ)
    environment["LMAFORMER_DATASET_ROOT"] = str(eval_root)
    result = subprocess.run([str(part) for part in command], cwd=str(REPO), env=environment)
    if result.returncode != 0:
        raise SystemExit("inference_swin.py failed with code {}".format(result.returncode))


def refresh_lmformer(teike_root, rows, args):
    dst = Path(args.lmformer_dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    (dst / "ImageSets").mkdir(parents=True, exist_ok=True)
    for row in rows:
        for kind in ("images", "masks"):
            link = dst / kind / row["seq"]
            target = teike_root / "data" / row["seq"] / kind
            if link.exists() or args.dry_run:
                continue
            if not target.is_dir():
                continue
            print("[联接] {} -> {}".format(link, target))
            create_directory_link(target, link)

    existing = set()
    val_file = dst / "ImageSets" / "val_new.txt"
    if val_file.is_file():
        existing = {line.strip() for line in val_file.read_text().splitlines() if line.strip()}
    added = [row["seq"] for row in rows if row["seq"] not in existing]
    if added:
        with open(val_file, "a", encoding="utf-8") as handle:
            for name in added:
                handle.write(name + "\n")
        print("[ImageSets] 追加到 val_new.txt: {}".format(", ".join(added)))
    else:
        print("[ImageSets] val_new.txt 已是最新，无新增")
    train_file = dst / "ImageSets" / "train_new.txt"
    if not train_file.is_file():
        train_file.write_text("", encoding="utf-8")
        print("[ImageSets] 建了空的 train_new.txt，训练前请自己填训练序列")
    print("[ImageSets] train_new.txt 现有: {}".format(
        ", ".join(line.strip() for line in train_file.read_text().splitlines() if line.strip()) or "(空)"))


def main():
    args = parse_args()
    teike_root = Path(args.teike_root)
    data_root, rows, pending = survey(teike_root, args.detect_out, args.force)
    print_survey(rows, pending)

    if args.add:
        print()
        add_pending(teike_root, pending, args)
    elif args.mask:
        print()
        export_masks(teike_root, rows, args)
    elif args.detect:
        print()
        detect_new(teike_root, rows, args)
    elif args.eval:
        print()
        evaluate(teike_root, rows, args)
    elif args.all:
        print()
        add_pending(teike_root, pending, args)
        export_masks(teike_root, rows, args)
        _, refreshed, _ = survey(teike_root, args.detect_out, args.force)
        detect_new(teike_root, refreshed, args)
        evaluate(teike_root, refreshed, args)
    elif args.lmformer:
        print()
        refresh_lmformer(teike_root, rows, args)
    else:
        print("\n（只做了体检。加 --add / --mask / --detect / --eval / --all / "
              "--lmformer 执行对应动作）")


if __name__ == "__main__":
    sys.exit(main())
