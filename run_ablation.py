"""MAFEM ablation driver for LMAFormer.

Trains and evaluates the four MAFEM variants requested for the ablation table:

    full         使用 MAFEM                      (CME + EnE)
    no_mafem     移除 MAFEM
    mafem_no_cme MAFEM 无 CME：仅保留 EnE 模块
    mafem_no_ene MAFEM 不带 EnE：仅保留 CME 模块

Each variant is trained from scratch with the Swin backbone and evaluated with
the standard `inference_swin.py` protocol, then the four metrics (IoU, nIoU, Pd,
Fa) are collected into `ablation_summary.csv` / `ablation_summary.md`.

Examples
--------
Sanity check that every variant builds and runs (a few seconds on GPU):

    python run_ablation.py --check

Full ablation suite on TSIRMT:

    python run_ablation.py --dataset_name TSIRMT --epochs 30 \
        --output_dir result_ablation --predict_dir predict_ablation

Re-evaluate checkpoints that were trained earlier:

    python run_ablation.py --dataset_name TSIRMT --skip_train
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# (variant, English description, label used in the paper table)
VARIANTS = [
    ('full',         'MAFEM (CME + EnE)',                   '使用 MAFEM'),
    ('no_mafem',     'MAFEM removed',                       '移除 MAFEM'),
    ('mafem_no_cme', 'MAFEM without CME (EnE only)',        'MAFEM 无 CME：仅保留 EnE 模块'),
    ('mafem_no_ene', 'MAFEM without EnE (CME only)',        'MAFEM 不带 EnE：仅保留 CME 模块'),
]
VARIANT_NAMES = [v[0] for v in VARIANTS]
VARIANT_LABELS = {v[0]: v[2] for v in VARIANTS}
VARIANT_DESCRIPTIONS = {v[0]: v[1] for v in VARIANTS}


def parse_args():
    parser = argparse.ArgumentParser('MAFEM ablation driver', add_help=True)
    parser.add_argument('--dataset_name', default='TSIRMT',
                        help='TSIRMT, NUDT-MIRSDT or IRDST')
    parser.add_argument('--variants', default=','.join(VARIANT_NAMES),
                        help='comma separated subset of: ' + ', '.join(VARIANT_NAMES))
    parser.add_argument('--output_dir', default='result_ablation',
                        help='training output root (one sub directory per variant)')
    parser.add_argument('--predict_dir', default='predict_ablation',
                        help='evaluation output root')
    parser.add_argument('--checkpoint', default='checkpoint_best.pth',
                        help='checkpoint file name produced by training')
    parser.add_argument('--python', default=sys.executable,
                        help='python interpreter used for train/inference subprocesses')
    parser.add_argument('--epochs', default=None, type=int,
                        help='override the training epochs')
    parser.add_argument('--train_size', default=None, type=int)
    parser.add_argument('--val_size', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--num_workers', default=None, type=int,
                        help='DataLoader workers; 0 is safest on Windows')
    parser.add_argument('--no_alloc_tuning', action='store_true',
                        help='do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '
                             '(it roughly halves the time per iteration of the two '
                             'MAFEM-heavy variants on a 12 GB card)')
    parser.add_argument('--check', action='store_true',
                        help='only build every variant and run one synthetic '
                             'forward/backward pass (no dataset needed)')
    parser.add_argument('--smoke', action='store_true',
                        help='run one real training iteration + one validation '
                             'sample per variant, then reload the checkpoint')
    parser.add_argument('--skip_train', action='store_true',
                        help='reuse existing checkpoints, only evaluate')
    parser.add_argument('--skip_infer', action='store_true',
                        help='only train, do not evaluate')
    parser.add_argument('--msc', action='store_true', default=True,
                        help='multi-scale evaluation (default: on)')
    parser.add_argument('--no_msc', dest='msc', action='store_false')
    parser.add_argument('--train_args', nargs=argparse.REMAINDER, default=[],
                        help='extra arguments forwarded to train_swin.py')
    parser.add_argument('--eval_args', nargs=argparse.REMAINDER, default=[],
                        help='extra arguments forwarded to inference_swin.py')
    return parser.parse_args()


def selected_variants(spec):
    names = [v.strip() for v in spec.split(',') if v.strip()]
    unknown = [v for v in names if v not in VARIANT_NAMES]
    if unknown:
        raise SystemExit('unknown variant(s): %s (expected %s)' % (', '.join(unknown), ', '.join(VARIANT_NAMES)))
    return names


def build_env(dataset_name):
    """Point the dataloader at ``Datasets/<dataset_name>`` unless the caller
    already set LMAFORMER_DATASET_ROOT."""
    env = os.environ.copy()
    if not env.get('LMAFORMER_DATASET_ROOT'):
        candidate = REPO_ROOT / 'Datasets' / dataset_name
        if candidate.is_dir():
            env['LMAFORMER_DATASET_ROOT'] = str(candidate)
    if not env.get('PYTORCH_CUDA_ALLOC_CONF'):
        # Measured on a 12 GB RTX 5070: the two MAFEM-heavy variants reserve
        # more than the card and spill into shared GPU memory; the expandable
        # segments allocator keeps that in check (~2x faster per iteration).
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return env


def run_command(cmd, env=None):
    print('\n>>> ' + ' '.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError('command failed with exit code %d: %s' % (result.returncode, ' '.join(cmd)))


def train_variant(args, variant):
    out_root = REPO_ROOT / args.output_dir
    exp_dir = out_root / args.dataset_name / ('ablation_' + variant)
    cmd = [args.python, 'train_swin.py',
           '--ablation', variant,
           '--dataset_name', args.dataset_name,
           '--output_dir', str(out_root),
           '--experiment_name', 'ablation_' + variant,
           '--is_train', '1']
    if args.epochs is not None:
        cmd += ['--epochs', str(args.epochs)]
    if args.train_size is not None:
        cmd += ['--train_size', str(args.train_size)]
    if args.val_size is not None:
        cmd += ['--val_size', str(args.val_size)]
    if args.batch_size is not None:
        cmd += ['--batch_size', str(args.batch_size)]
    if args.num_workers is not None:
        cmd += ['--num_workers', str(args.num_workers)]
    cmd += list(args.train_args)
    run_command(cmd, env=build_env(args.dataset_name))
    return exp_dir / args.checkpoint


def eval_variant(args, variant, checkpoint):
    eval_dir = REPO_ROOT / args.predict_dir / variant
    cmd = [args.python, 'inference_swin.py',
           '--ablation', variant,
           '--dataset', args.dataset_name,
           '--model_path', str(checkpoint),
           '--output_dir', str(eval_dir),
           '--num_workers', str(args.num_workers if args.num_workers is not None else 0)]
    if args.val_size is not None:
        cmd += ['--val_size', str(args.val_size)]
    if args.batch_size is not None:
        cmd += ['--batch_size', str(args.batch_size)]
    if args.msc:
        cmd += ['--msc']
    cmd += list(args.eval_args)
    run_command(cmd, env=build_env(args.dataset_name))
    return eval_dir


def read_metrics(eval_dir, dataset_name):
    """Read IoU/nIoU/Pd/Fa written by MFIRSTD.evals."""
    metrics = {'IoU': None, 'nIoU': None, 'Pd': None, 'Fa': None}
    mat_path = Path(eval_dir) / dataset_name / 'LMAFormer_PD_FA_255.mat'
    if mat_path.exists():
        from scipy.io import loadmat
        mat = loadmat(str(mat_path))
        for key, out_key in (('IOU', 'IoU'), ('nIoU', 'nIoU'), ('PD0', 'Pd'), ('FA0', 'Fa')):
            if key in mat:
                metrics[out_key] = float(mat[key].ravel()[0])
    else:
        print('warning: %s not found' % mat_path)
    csv_path = Path(eval_dir) / 'dataset_results.csv'
    if metrics['IoU'] is None and csv_path.exists():
        rows = list(csv.reader(open(csv_path)))
        if len(rows) >= 2:
            metrics['IoU'] = float(rows[1][-1])
    return metrics


def write_summary(args, rows):
    header = ['variant', 'description', 'label', 'IoU', 'nIoU', 'Pd', 'Fa', 'status']
    out_root = REPO_ROOT / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / 'ablation_summary.csv'
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(h, '') if row.get(h) is not None else '' for h in header])

    def fmt(value):
        return '-' if value is None else '%0.2f' % value

    md_path = out_root / 'ablation_summary.md'
    with open(md_path, 'w') as handle:
        handle.write('| 设置 | 说明 | IoU | nIoU | Pd | Fa |\n')
        handle.write('| --- | --- | --- | --- | --- | --- |\n')
        for row in rows:
            handle.write('| %s | %s | %s | %s | %s | %s |\n' % (
                row['label'], row['description'], fmt(row['IoU']), fmt(row['nIoU']),
                fmt(row['Pd']), fmt(row['Fa'])))
    print('\nsummary written to:\n  %s\n  %s' % (csv_path, md_path))
    print(open(md_path).read())


def run_check(args):
    """Build each variant and run one synthetic forward/backward pass."""
    import torch
    from MFIRSTD.utils.misc import NestedTensor
    from MFIRSTD.models.lmaf_swin import build_model_lmaf_swinbackbone
    from inference_swin import get_args_parser as get_infer_parser

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames, size = 5, 224
    for variant in selected_variants(args.variants):
        parser = get_infer_parser()
        cli = ['--ablation', variant, '--device', str(device), '--num_frames', str(frames)]
        model_args = parser.parse_args(cli)
        model_args.is_train = 0
        model_args.aux_loss = 0
        model_args.aux_loss_norm = 0
        model, criterion = build_model_lmaf_swinbackbone(model_args)
        model.to(device).train()
        samples = NestedTensor(torch.randn(frames, 3, size, size, device=device),
                               torch.zeros(frames, size, size, dtype=torch.bool, device=device))
        targets = [{'masks': torch.zeros(frames, size, size, device=device)}]
        outputs, outputs_ori, flow_f, flow_b = model(samples)
        loss = criterion(outputs, targets, samples.tensors, flow_f, flow_b)
        loss_ori = criterion(outputs_ori, targets, samples.tensors, flow_f, flow_b)
        total = sum(loss.values()) + 0.1 * sum(loss_ori.values())
        total.backward()
        n_params = sum(p.numel() for p in model.parameters())
        n_flow = sum(p.numel() for n, p in model.named_parameters() if n.startswith('sc_4.'))
        print('[check] %-13s masks=%s flows=%s params=%.2fM mafem=%.2fM losses=%s'
              % (variant, tuple(outputs['pred_masks'].shape), len(flow_f), n_params / 1e6,
                 n_flow / 1e6, sorted(loss.keys())))
        model.cpu()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    print('\nall variants built, forward/backward OK')


def run_smoke(args):
    """One real training iteration + one validation sample per variant.

    This is the cheap end-to-end gate to run before the real ablation suite: it
    exercises the dataloader, the criterion (including the flow loss being
    present or absent), the optimizer step and reloading the checkpoint with
    the very same command line the evaluation stage uses.
    """
    # path_config reads this variable at import time, so it has to be set
    # before the dataset modules are imported.
    dataset_root = build_env(args.dataset_name).get('LMAFORMER_DATASET_ROOT')
    if dataset_root:
        os.environ['LMAFORMER_DATASET_ROOT'] = dataset_root
    import torch
    from torch.utils.data import DataLoader
    import train_swin
    from MFIRSTD.datasets.test.val_data import ValDataset
    from MFIRSTD.datasets.train.train_data import TrainDataset
    from MFIRSTD.evals import infer_on_dataset_train
    from MFIRSTD.models.lmaf_swin import build_model_lmaf_swinbackbone
    from MFIRSTD.utils import misc

    out_root = REPO_ROOT / args.output_dir / 'viz'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def make_args(variant):
        parser = train_swin.get_args_parser()
        parsed = parser.parse_args([
            '--ablation', variant,
            '--dataset_name', args.dataset_name,
            '--train_size', '128', '--val_size', '128',
            '--epochs', '1', '--num_workers', '0',
            '--output_dir', str(REPO_ROOT / args.output_dir),
            '--experiment_name', 'smoke_' + variant,
        ])
        parsed.pretrain_settings = {}
        parsed.output_dir = str(REPO_ROOT / args.output_dir)
        return parsed

    train_ds = TrainDataset(num_frames=5, train_size=128)
    val_ds = ValDataset(num_frames=5, val_size=128)
    # MAFEM treats the whole batch dimension as the temporal axis, so the
    # training batch size must stay 1.
    train_loader = DataLoader(train_ds, batch_sampler=[[0]], collate_fn=misc.collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_sampler=[[0]], collate_fn=misc.collate_fn, num_workers=0)
    for variant in selected_variants(args.variants):
        model_args = make_args(variant)
        model, criterion = build_model_lmaf_swinbackbone(model_args)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        viz_dir = out_root / variant
        viz_dir.mkdir(parents=True, exist_ok=True)
        stats = train_swin.train_one_epoch(model, criterion, train_loader, optimizer, device, 0,
                                           model_args.clip_max_norm, viz_dir, use_wandb=False,
                                           viz_freq=-1, total_epochs=1, args=model_args)
        mean_iou = infer_on_dataset_train(model, val_loader, device, msc=False, flip=False,
                                          save_pred=False, out_dir=str(viz_dir / 'val'))
        ckpt = out_root / ('smoke_%s.pth' % variant)
        torch.save({'model': model.state_dict()}, ckpt)
        reload_args = make_args(variant)
        reload_args.is_train = 0
        reload_model, _ = build_model_lmaf_swinbackbone(reload_args)
        reload_model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'],
                                     strict=True)
        loss_keys = sorted(k for k in stats if k.startswith('loss') and 'unscaled' not in k)
        print('[smoke] %-13s loss=%.4f val_mIoU=%.4f checkpoint reload OK losses=%s'
              % (variant, stats['loss'], mean_iou, loss_keys))
        del model, reload_model, optimizer, criterion
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('\nsmoke training OK for all variants')


def main():
    args = parse_args()
    if args.check:
        run_check(args)
        return 0
    if args.smoke:
        run_smoke(args)
        return 0
    if args.batch_size not in (None, 1):
        raise SystemExit('MAFEM treats the batch dimension as the temporal axis, '
                         'so --batch_size has to stay 1 (got %d).' % args.batch_size)

    rows = []
    failures = []
    for variant in selected_variants(args.variants):
        row = {'variant': variant, 'label': VARIANT_LABELS[variant],
               'description': VARIANT_DESCRIPTIONS[variant], 'status': 'OK'}
        rows.append(row)
        try:
            out_root = REPO_ROOT / args.output_dir
            checkpoint = out_root / args.dataset_name / ('ablation_' + variant) / args.checkpoint
            if not args.skip_train:
                checkpoint = train_variant(args, variant)
            if not args.skip_infer:
                if not Path(checkpoint).exists():
                    raise RuntimeError('checkpoint not found: %s' % checkpoint)
                eval_dir = eval_variant(args, variant, checkpoint)
                row.update(read_metrics(eval_dir, args.dataset_name))
        except Exception as error:  # keep going, report at the end
            row['status'] = 'FAILED'
            failures.append((variant, str(error)))
            print('!!! variant %s failed: %s' % (variant, error))
    write_summary(args, rows)
    if failures:
        print('\nfailed variants:')
        for variant, error in failures:
            print('  %s: %s' % (variant, error))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
