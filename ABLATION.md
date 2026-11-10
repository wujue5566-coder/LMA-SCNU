# MAFEM 消融实验使用说明

本文件说明怎么在本仓库跑 MAFEM 消融实验。命令都在你的机器上实测过（RTX 5070 12GB，Windows）。
**先读第 3 节的显存实测，再决定用哪条命令。**

## 1. 四组设置与开关

| 论文表格设置 | `--ablation` | 代码行为 | 参数量 | 光流损失 |
| --- | --- | --- | --- | --- |
| 使用 MAFEM | `full` | CME + EnE 全开 | 407.7M | 有 |
| 移除 MAFEM | `no_mafem` | 不建 MAFEM，Transformer 不乘 `sc_weight` | 102.7M | 无 |
| MAFEM 无 CME（仅 EnE） | `mafem_no_cme` | 去掉光流估计分支，可变形注意力用纯学习偏移 | 406.4M | 无 |
| MAFEM 不带 EnE（仅 CME） | `mafem_no_ene` | 保留光流做运动补偿，去掉可变形注意力 | 104.9M | 有 |

MAFEM = [`SC_Module.py`](MFIRSTD/models/MAFEM_block/SC_Module.py) 里的 `SC` 模块。
其中 **CME = 光流估计分支**（`compute_flow`），**EnE = 光流引导的可变形注意力**
（`DeformableAttnBlock`）。`collect_eng/bn_last/sigmoid` 是三个 MAFEM 变体共用的输出头，
不保留它模块就产不出 Transformer 需要的 `sc_weight`。

## 2. 三步走：先自检，再试跑，最后正式跑

先设定解释器（编译好的可变形注意力算子只在这个环境能加载）：

```powershell
cd D:\PycharmProjects\LMAFormer
$py = "D:\anaconda3\envs\IRSTD\python.exe"
& $py -c "import torch, MultiScaleDeformableAttention as m; print('OK', torch.__version__)"
```

### 步骤 1：自检（几十秒）

```powershell
& $py run_ablation.py --check
```

四组都能前向 + 反向就说明代码没问题，预期输出：

```
[check] full          masks=(1, 5, 28, 28) flows=1 params=407.67M mafem=304.97M
[check] no_mafem      masks=(1, 5, 28, 28) flows=0 params=102.69M mafem=0.00M
[check] mafem_no_cme  masks=(1, 5, 28, 28) flows=0 params=406.43M mafem=303.74M
[check] mafem_no_ene  masks=(1, 5, 28, 28) flows=1 params=104.85M mafem=2.16M
```

再确认真实数据链路（1 次迭代 + 1 次验证 + 权重回载，约 1 分钟）：

```powershell
& $py run_ablation.py --smoke --dataset_name IRDST
```

### 步骤 2：先跑一组轻量变体，确认端到端

```powershell
& $py run_ablation.py --dataset_name IRDST --variants no_mafem --epochs 1 --train_size 288 --val_size 288
```

跑完检查 `result_ablation\IRDST\ablation_no_mafem\` 下有 `checkpoint_best.pth`，
`result_ablation\ablation_summary.md` 里出现这一行。

### 步骤 3：正式跑四组（顺序执行，不是同时跑）

```powershell
# IRDST，288px，四组约 22 h
& $py run_ablation.py --dataset_name IRDST --epochs 30 --train_size 288 --val_size 288 --output_dir result_ablation --predict_dir predict_ablation

# TSIRMT，288px，四组约 6.4 天
& $py run_ablation.py --dataset_name TSIRMT --epochs 30 --train_size 288 --val_size 288 --output_dir result_ablation --predict_dir predict_ablation
```

脚本对四个变体**依次**执行：训练 → 用同一变体配置推理 → 读取 IoU/nIoU/Pd/Fa → 写汇总表。
每个变体是独立进程，跑完显存完全释放，**不存在四组显存叠加**。

想恢复论文的 400px 协议，去掉 `--train_size 288 --val_size 288` 即可，但看第 3 节就知道
12GB 显卡扛不住。

## 3. 显存实测（RTX 5070，11.94 GiB，5 帧，batch=1）

格式：`稳态 reserved / 单步耗时`（单步耗时已开启 `expandable_segments`）：

| 变体 | 400px（论文协议） | 320px | 288px | 256px |
| --- | --- | --- | --- | --- |
| `full` | 14.3 GiB / 4.0 s（溢出） | 11.2 GiB / 0.43 s | 11.6 GiB / 0.47 s | 10.7 GiB / 0.25 s |
| `mafem_no_cme` | 13.3 GiB / 3.1 s（溢出） | 12.1 GiB / 1.65 s（刚超） | 11.5 GiB / 0.45 s | 10.7 GiB / 0.24 s |
| `no_mafem` | 8.4 GiB / 0.30 s | 6.1 GiB / 0.20 s | - | - |
| `mafem_no_ene` | 10.1 GiB / 0.37 s | 6.0 GiB / 0.20 s | - | - |

**288px 是四个变体都不溢出显存的最大分辨率**（11.94 GiB 卡上两组重变体都是 11.5 GiB 左右），
所以四组统一用 `--train_size 288 --val_size 288` 是最省时间又不失真的档位。

怎么读这张表：

* **只有 `full` 和 `mafem_no_cme` 会超显存**（407M 参数，参数+梯度+Adam 状态本身就占 6.5 GiB）。
  超过 12GB 时 Windows 会借用共享显存，**不会直接 OOM，但速度掉 8 倍左右**。
* 慢的主因不是"算得多"而是"溢出"：400px 下 `full` 第一步只要 1.1 s，开始借共享显存后变成 8.5 s。
  脚本默认设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，实测把这两个重变体
  从 8.5 s/步 降到 4.0 s/步（想去掉用 `--no_alloc_tuning`）。
* 把分辨率降到 288/320 后完全装进显存，重变体从 4 s/步 降到 0.45 s/步，快近 9 倍。
  四组都用同一个分辨率时比较是公平的，但和论文 400px 的数字不能直接对比，需要在论文里写明。
* 显存不够时的处理顺序：先关掉浏览器/PyCharm 等占用显存的程序 → 降 `--train_size` →
  换显存更大的机器。`--batch_size` 不能用来省显存（必须是 1，见第 7 节）。

**显存和数据集无关**：IRDST 和 TSIRMT 的显存表现完全一样（同一张卡、同一个变体、
同样的分辨率 ⇒ 同样的显存）。数据集只决定训练帧数，也就是只影响**时间**——
所以第 4 节里 TSIRMT 更贵是"时间贵"，不是"显存贵"。

推理/评估阶段的显存比训练低得多（没有梯度），实测 400px + flip + msc：

| 变体 | 推理 peak reserved |
| --- | --- |
| `full` | 3.03 GiB |
| `mafem_no_ene` | 1.52 GiB |

四个变体评估都不会有显存问题，`--msc` 只是把多尺度前向串行跑，峰值不叠加。

## 4. 时间和磁盘估算（30 epoch）

训练样本数：IRDST 2000 帧/epoch，TSIRMT 14000 帧/epoch（每帧是一个 5 帧样本）。

| 数据 / 分辨率 | `full` | `mafem_no_cme` | `no_mafem` | `mafem_no_ene` | 四组合计 |
| --- | --- | --- | --- | --- | --- |
| IRDST @400px | 67 h | 52 h | 5 h | 6 h | 约 5.4 天 |
| IRDST @288px | 7.8 h | 7.5 h | 3.3 h | 3.3 h | **约 22 h** |
| IRDST @256px | 4.2 h | 4.0 h | 3 h | 3 h | 约 14 h |
| TSIRMT @400px | 19 天 | 15 天 | 1.5 天 | 1.8 天 | 不可行 |
| TSIRMT @288px | 55 h | 52 h | 23 h | 23 h | 约 6.4 天 |
| TSIRMT @256px | 29 h | 28 h | 21 h | 21 h | 约 4 天 |

这只算训练，不含每个 epoch 的验证（验证集 IRDST 100 帧、TSIRMT 6000 帧，还带 flip 增强；
TSIRMT@400px 光验证每 epoch 就要约 1 小时）。

磁盘：`full` 单个 checkpoint 约 1.63 GB，每 epoch 会同时存 `checkpoint_last.pth` 和
`checkpoint_best.pth`，四组合计约 **8~9 GB**，外加 `predict_ablation` 下保存的预测图。

### 想最快拿到整张表，按这个顺序省时间

1. **四组统一用 `--train_size 288 --val_size 288`**：四组显存都够，比 400px 快约 6 倍。
2. **`使用 MAFEM` 那一行复用官方权重**（见第 5 节）：IRDST 省 8 h，TSIRMT 省 55 h。
3. **先跑 IRDST**：TSIRMT 训练帧数是 IRDST 的 7 倍，时间也差不多 7 倍。
4. 还想更快就再降到 256px 或减少 `--epochs`（IRDST@256 约 14 h）。

再往下压就必须牺牲可比性了（四组的分辨率、epoch、数据划分必须完全一致，
否则消融表不成立），那种情况下要在论文里写明改动。

## 5. 省时间：`使用 MAFEM` 这一行可以直接复用官方权重

已验证：`result\IRDST\checkpoint_IRDST.pth` 与 `result\TSIRMT\checkpoint_TSIRMT.pth`
都能被 `--ablation full` 严格加载（missing=0、unexpected=0），它们就是 `full` 模型。

```powershell
New-Item -ItemType Directory -Force result_ablation\IRDST\ablation_full | Out-Null
Copy-Item result\IRDST\checkpoint_IRDST.pth result_ablation\IRDST\ablation_full\checkpoint_best.pth
& $py run_ablation.py --dataset_name IRDST --variants full --skip_train --val_size 400
```

这样可以省掉 `full` 那组几十小时训练，但口径要注意：官方权重是 400px + 作者训练配方，
如果你另外三组用 320px 重训，两者严格来说不可比。三种口径：

1. 复用官方 `full`，其余三组也用 400px 训（IRDST 约 3 天）；
2. 四组全部 288px 从头训（IRDST 约 22 h，口径一致，论文里注明分辨率）；**推荐**
3. 四组全部 400px 从头训（需要 24GB 以上显存，建议租云卡）。

## 6. 结果在哪

* 训练产物：`result_ablation\<数据集>\ablation_<变体>\checkpoint_best.pth`、`out.log`
* 评估产物：`predict_ablation\<变体>\<数据集>\LMAFormer_PD_FA_255.mat`（含 `IOU/nIoU/PD0/FA0`）
  以及 `predict_ablation\<变体>\dataset_results.csv`（逐序列 IoU）
* 汇总表：`result_ablation\ablation_summary.md`（可直接贴论文）和 `ablation_summary.csv`

只重跑评估（比如换了 `--val_size`）：

```powershell
& $py run_ablation.py --dataset_name IRDST --skip_train --val_size 400
```

## 7. 常见问题

必须先知道的两条硬约束：

* `--batch_size` 必须为 1：MAFEM 把 batch 维当时序维用（脚本已拦截大于 1 的情况）。
* `--num_frames` 必须为 5：MAFEM 里光流与 5 帧特征是写死的（脚本会直接报错提示）。

推理必须带与训练相同的 `--ablation`，否则网络结构不一致、权重加载失败。
不要拿同一份权重切换 `--ablation` 去"造"消融结果。

`DLL load failed` / 找不到算子：用错解释器了。只有
`D:\anaconda3\envs\IRSTD\python.exe`（torch 2.11+cu128）能加载
`MFIRSTD\models\ops\MultiScaleDeformableAttention.cp310-win_amd64.pyd`；
`LMAFormer` 环境（torch 2.8）需要在 `MFIRSTD\models\ops` 下执行
`python setup.py build_ext --inplace` 重新编译。

每个变体都要重新训练：`--ablation` 改的是网络结构与损失项，除此之外的超参
（epoch、学习率、数据划分、随机种子）四组必须一致。

缺少 `ckpts\resnet_init\384_coco_r101.pth`：训练会打印警告并改用默认初始化。
四组一视同仁，比较仍然公平；想让训练与原仓库完全一致，把该权重补到对应目录即可。

中断后继续：脚本不支持断点续训，直接重跑对应变体即可
（例如 `--variants mafem_no_cme` 只跑一组）。

## 8. 参数速查

```powershell
& $py run_ablation.py --help
```

| 参数 | 作用 |
| --- | --- |
| `--check` | 只做四组模型的构建 + 前向/反向自检 |
| `--smoke` | 每组跑 1 次真实训练迭代 + 1 次验证 + 权重回载 |
| `--variants a,b` | 只跑指定变体：`full,no_mafem,mafem_no_cme,mafem_no_ene` |
| `--epochs` / `--train_size` / `--val_size` / `--batch_size` / `--num_workers` | 透传给训练脚本 |
| `--skip_train` | 复用已有 checkpoint，只评估 |
| `--skip_infer` | 只训练不评估 |
| `--no_msc` | 评估时关闭多尺度增强 |
| `--no_alloc_tuning` | 不设置 `PYTORCH_CUDA_ALLOC_CONF` |
| `--train_args ...` / `--eval_args ...` | 额外参数透传给 `train_swin.py` / `inference_swin.py` |

数据集由环境变量 `LMAFORMER_DATASET_ROOT` 决定，脚本未设置时会自动指向
`Datasets\<dataset_name>`（`--dataset_name` 本身只影响输出目录命名）。

## 9. 手动单条命令（等价写法）

```powershell
$env:LMAFORMER_DATASET_ROOT = "D:\PycharmProjects\LMAFormer\Datasets\IRDST"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# 训练（每个变体跑一次，不要并行）
python train_swin.py --ablation mafem_no_cme --dataset_name IRDST --train_size 288 --val_size 288 --output_dir result_ablation --experiment_name ablation_mafem_no_cme

# 评估（--ablation 必须与训练一致）
python inference_swin.py --ablation mafem_no_cme --dataset IRDST --model_path result_ablation\IRDST\ablation_mafem_no_cme\checkpoint_best.pth --val_size 288 --flip --msc --output_dir predict_ablation\mafem_no_cme
```
