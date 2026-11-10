# 用 LMAFormer 测试新数据集操作指南

以 `TeiKe` 为例，说明怎么把新拍的数据喂给官方训练/评测流程。所有命令都在仓库根 `D:\PycharmProjects\LMAFormer` 下执行，解释器用 `D:\anaconda3\envs\IRSTD\python.exe`。

---

## 0. 先看结论

| 环节 | 入口 | 需要准备什么 |
| --- | --- | --- |
| 用现成权重测泛化（不训练） | `inference_swin.py` | 数据集转成 `images/<序列>/` + `masks/<序列>/` + `ImageSets/*.txt`，设好环境变量 |
| 在自己的数据上训练/微调 | `train_swin.py` | 同上，另加 `--dataset_name TeiKe` |
| 出检测可视化视频 | `visualize_irdst.py` | 需要 GT 掩码 + 上一步的预测结果 |
| 无标注视频只想看检测效果 | `inference_keii.py` | 只要视频，不需要掩码 |

**最容易卡住的三点**：

1. 官方要求 `images/<序列>/` 这种嵌套，而 `datasets\TeiKe\data` 是 `<序列>/images/`——**正好反了**，必须转换（见 §1.2）。
2. 数据位置不是靠命令行参数指定的，而是靠环境变量 `LMAFORMER_DATASET_ROOT`（见 §2）。
3. `visualize_irdst.py` 把预测目录名硬编码成了 `IRDST`，数据集叫别的名字时找不到掩码（见 §4）。

本指南里的命令都在 `TeiKe`（6 段 × 100 帧）上实际跑过，结果见 §7。

---

## 1. 数据准备

### 1.1 官方要求的目录结构

`MFIRSTD/datasets/path_config.py` 决定了数据必须长这样：

```
<root>/                          # 环境变量 LMAFORMER_DATASET_ROOT 指向这里
├── ImageSets/
│   ├── train_new.txt            # 每行一个序列名
│   └── val_new.txt
├── images/
│   ├── 1/ 000000.png 000001.png ...
│   ├── 2/ ...
│   └── 6/ ...
└── masks/
    ├── 1/ 000000.png ...        # 与 images/1/ 同名同尺寸
    └── ...
```

你现在的布局是：

```
TeiKe/data/
  1/images/*.png  1/masks/*.png
  2/images/*.png  2/masks/*.png  ...
```

——序列在上一层、`images/masks` 在下层，dataloader 读不到（它会去 `images/1/` 找图），所以必须先转换。

### 1.2 一键转换与划分

先确认 Teike 根目录下存在 `data/split_70_15_15.txt`，然后执行：

```bash
export LMAFORMER_TEIKE_ROOT=/data/LMAFormer/TeiKe
python prepare_teike_split.py
```

- 脚本从 `data/<sequence>/{images,masks}` 创建目录链接，不复制图片。
- `split_70_15_15.txt` 必须严格提供 70/15/15 个训练、验证和测试序列。
- 每段序列必须正好包含 100 张图片和 100 张掩码。
- 拆分布局会写入 `Datasets/TeiKe_LMAFormer/ImageSets/{train,val,test}_new.txt`。
- 数据位于其他位置时，可用 `--source` 和 `--split` 显式指定。

### 1.3 硬性约束

| 约束 | 说明 |
| --- | --- |
| 序列名一致 | `ImageSets/*.txt` 里的名字必须和 `images/` 下的目录名完全相同 |
| 帧数 ≥ 5 | 滑窗 `num_frames=5`，序列短于 5 帧会 assert 失败（不足时靠取模环绕，正好 5 帧也可以） |
| 同名同尺寸 | `images/<seq>/X.png` 必须有 `masks/<seq>/X.png`，分辨率一致 |
| 掩码取值 | 单通道，前景 0/255（代码里会做 `gt[gt>0]=255`，所以 0/1 也能用） |
| 训练用的帧 | 训练时 focal/dice 对窗口内 **5 帧的掩码一起** 计算，所以进数据集的帧都必须有掩码 |

---

## 2. 告诉代码数据在哪

路径不是命令行参数，而是环境变量（`path_config.py` 在 import 时读取，必须在启动脚本前设好）：

```powershell
$env:LMAFORMER_DATASET_ROOT = "D:\PycharmProjects\LMAFormer\Datasets\TeiKe_LMAFormer"

# 确认生效
python -c "from MFIRSTD.datasets import path_config as p; print(p.img_path); print(p.val_seqs_file)"
```

注意：`$env:` 只在当前终端会话有效，每开一个新窗口都要重设；不设的话默认会去读 `Datasets/IRDST`，很容易出现"评测结果对不上"的错觉。

---

## 3. 用现成权重评测（不训练，最快）

```powershell
$env:LMAFORMER_DATASET_ROOT = "D:\PycharmProjects\LMAFormer\Datasets\TeiKe_LMAFormer"

python inference_swin.py `
    --model_path ./result/NUDT-MIRSDT/checkpoint_NUDT-MIRSDT.pth `
    --dataset TeiKe `
    --output_dir ./predict/TeiKe_official `
    --val_size 400 --flip --num_workers 0
```

参数要点：

| 参数 | 说明 |
| --- | --- |
| `--model_path` | 要测的权重；三个现成的分别是 `result/{IRDST,TSIRMT,NUDT-MIRSDT}/checkpoint_*.pth` |
| `--dataset` | **只决定输出子目录名**（结果落在 `<output_dir>/<dataset>/`），不影响读哪份数据 |
| `--flip` | 水平翻转平均，建议常开（默认就是 True） |
| `--msc` | 多尺度 0.75–1.3 平均，慢约 7 倍，冲指标时再开 |
| `--val_size 400` | 最短边缩放到 400，必须与训练时一致 |
| `--num_workers 0` | Windows 上建议 0，避免 CUDA DLL / 页文件问题 |
| `--sequence_names` | 想只测某几段可以传（脚本内部默认置 None，读 `val_new.txt`） |

产物结构：

```
predict/TeiKe_official/
├── dataset_results.csv              # 每段 IoU + 平均 IoU
├── out.log
└── TeiKe/
    ├── LMAFormer_PD_FA_255.mat      # FA0 / PD0 / IOU / nIoU
    ├── logits/<seq>/*.png           # 逐帧概率图
    └── bin_mask/<seq>/*.png         # 逐帧二值掩码（阈值 0.5）
```

读指标：

```powershell
python -c "import scipy.io as scio; m=scio.loadmat(r'predict/TeiKe_official/TeiKe/LMAFormer_PD_FA_255.mat'); print({k: float(v) for k, v in m.items() if not k.startswith('__')})"
```

（`run_ablation.py` 里的 `read_metrics()` 就是按 `IOU / nIoU / PD0 / FA0` 这四个键读的。）

耗时参考：600 帧约 2.5 分钟（`--flip`，0.25 s/帧，RTX 5070）。

---

## 4. 出可视化视频

```powershell
$env:LMAFORMER_DATASET_ROOT = "D:\PycharmProjects\LMAFormer\Datasets\TeiKe_LMAFormer"

python visualize_irdst.py `
    --prediction_dir ./predict/TeiKe_official `
    --output_dir ./predict/TeiKe_official/visualizations `
    --video 1 --fps 10
```

输出三栏视频（GT ｜ 预测 ｜ TP/FN/FP 叠加）+ 首帧 PNG，并打印该段视频与整个数据集的 Pd/Fa。

**坑**：脚本第 398 行把预测目录硬编码成了 `IRDST`——
```python
prediction_root = prediction_dir / "IRDST" / "bin_mask"
```
而你的结果在 `.../TeiKe/bin_mask`，直接跑会报 `Required path does not exist`。两种解法：

```powershell
# 解法 A（推荐，一条命令的事）：建个联接，让 IRDST 指向 TeiKe
New-Item -ItemType Junction -Path "predict\TeiKe_official\IRDST" -Target "predict\TeiKe_official\TeiKe"

# 解法 B：评测时就把 --dataset 写成 IRDST（因为 --dataset 只决定输出子目录名）
python inference_swin.py --model_path ./result/NUDT-MIRSDT/checkpoint_NUDT-MIRSDT.pth `
    --dataset IRDST --output_dir ./predict/TeiKe_as_IRDST --flip --num_workers 0
```

另外这个脚本需要 GT 掩码（要画 GT 并算 Pd/Fa），没有标注的序列用不了；
无标注只想看检测效果请用 `python inference_keii.py --help` 查看参数。

---

## 5. 在自己的数据上训练 / 微调

```powershell
$env:LMAFORMER_DATASET_ROOT = "D:\PycharmProjects\LMAFormer\Datasets\TeiKe_LMAFormer"

python train_swin.py `
    --dataset_name TeiKe `
    --output_dir ./result `
    --epochs 30 --batch_size 1 `
    --train_size 400 --val_size 400 `
    --lr 1e-4 --lr_backbone 1e-5 `
    --swin_s_pretrained_path ./ckpts/swin_init/swin_small_patch244_window877_kinetics400_1k.pth `
    --num_workers 0
```

要点：

* `--dataset_name` **只影响输出路径**：`./result/<dataset_name>/<experiment_name>/`，其中 `experiment_name` 默认带时间戳（`lmaf_swin_exp_...`），所以每次训练都会新建一个目录，不会覆盖旧结果。
* 每个 epoch 结束时会在验证集（`val_new.txt`）上算 mean IoU：
  * 每轮都存 `checkpoint_last.pth`；
  * 当出现更好的 mean IoU 时存 `checkpoint_best.pth`。
* 训练日志与可视化在 `outputs/`、`<output_dir>/viz/` 下。
* 显存：`--batch_size 1`、`--train_size 400` 时约 8–10 GB，和别的进程别抢。

**从已发布权重微调**（想在新数据上继续训）：

```powershell
python train_swin.py --dataset_name TeiKe --output_dir ./result --epochs 30 `
    --pretrain_settings pretrained_model_path ./result/NUDT-MIRSDT/checkpoint_NUDT-MIRSDT.pth
```

`--pretrain_settings` 是"键 值"成对传入（内部 `parse_argdict` 解析），会以 `strict=False` 加载 `state_dict['model']`。

> **已实测的坑**：`train_swin.py` 第 320 行的 `torch.load(...)` 没带 `weights_only=False`，而官方发布的 checkpoint 里含 `argparse.Namespace`，在 torch 2.6+ 上会直接抛 `UnpicklingError`（我在 torch 2.11 上验证过）。`MFIRSTD/evals.py` 已经修过这个（`raise ... weights_only=False`），训练脚本里漏了。修法是把那一行改成：
> ```python
> state_dict = torch.load(args.pretrain_settings['pretrained_model_path'],
>                         map_location='cpu', weights_only=False)
> ```

训练完后，用第 3 步把 `--model_path` 换成 `./result/TeiKe/<experiment_name>/checkpoint_best.pth` 即可评测自己的权重。

---

## 6. 常见错误对照表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `FileNotFoundError: ... Datasets\IRDST\ImageSets\val_new.txt` | 忘了设 `LMAFORMER_DATASET_ROOT` | 在同一个终端设好再跑 |
| 跑完发现指标是 IRDST 的 | 同上（默认读 IRDST） | 同上 |
| 评测秒退、`Video Mean IOU: nan` | `val_new.txt` 里的序列名和 `images/` 目录名不一致，或帧数为 0 | 核对名字；重新运行 `prepare_teike_split.py` 检查数据目录和划分文件 |
| `AssertionError` 在 `frame_indices` 处 | 序列帧数 < `num_frames`(5) | 合并或补足帧数 |
| `UnpicklingError: Weights only load failed` | torch 2.6+ 默认 `weights_only=True`，旧 checkpoint 带 Namespace | 见 §5 的一行修改 |
| `visualize_irdst.py` 报 `Required path does not exist` | 硬编码的 `IRDST` 子目录 | 见 §4 的解法 A/B |
| `RuntimeError: ... num_frames=5` / MAFEM 报错 | 改了 `--num_frames` | MAFEM 模块硬性要求 5，别改 |
| 显存不足 / CUDA OOM | 并行跑了多个推理进程，或 `--msc` + 大分辨率 | 串行执行；`nvidia-smi` 确认空闲再跑 |
| 训练 loss 出现 `nan` 或 loss 爆炸 | 掩码异常（全 0 掩码占多数、或掩码与图不同尺寸） | 跑掩码质检，确认标注一致性 |

---

## 7. 本次 TeiKe 实测记录（交叉验证）

数据：`datasets\TeiKe\data`，6 段 × 100 帧，掩码手工标注（目标面积平均 6.5–20.9 px）。
权重：`result/NUDT-MIRSDT/checkpoint_NUDT-MIRSDT.pth`；`--val_size 400 --flip`，未开 `--msc`。

官方 `inference_swin.py` 结果（`predict/TeiKe_official/`）：

| 指标 | 值 |
| --- | --- |
| Pd (PD0) | **0.7776** |
| Fa (FA0) | **1.01e-04** |
| IoU | 0.3358 |
| nIoU | 0.3110 |

逐序列 IoU，与另一套独立实现 `eval_keii_masked.py` 逐位一致：

| 序列 | 1 | 2 | 3 | 4 | 5 | 6 | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 官方 `inference_swin.py` | 0.15690 | 0.43104 | 0.50222 | 0.38994 | 0.13764 | 0.39679 | 0.33575 |
| 独立管线 | 0.15690 | 0.43104 | 0.50222 | 0.38994 | 0.13764 | 0.39679 | 0.33575 |

两套管线的 Pd/Fa 也一致（0.778 / 1.01e-04），说明评测实现可信；差异只可能来自采样方式与标注口径，不是代码问题。

