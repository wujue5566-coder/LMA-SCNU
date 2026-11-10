## PyTorch implementation of "LMAFormer: Local Motion Aware Transformer for Small Moving Infrared Target Detection" 
[Project](https://github.com/lifier/LMAFormer) - [Paper](https://ieeexplore.ieee.org/document/10758760)
<hr>

# Abstract
<p align="justify">
In temporal infrared small target detection, it is crucial to leverage the disparities in spatiotemporal characteristics between the target and the background to distinguish the former. However, remote imaging and the relative motion between the detection platform and the background cause significant coupling of spatiotemporal characteristics, making target detection highly challenging. To address these challenges, we propose a network named LMAFormer. First, we introduce a local motion-aware spatiotemporal attention mechanism that aligns and enhances multiframe features to extract local spatiotemporal salient features of targets while avoiding interference from moving backgrounds. Second, we employ a multiscale fusion transformer encoder that computes self-attention weights across and within scales during encoding, to establish multiscale correlations among different regions of temporal images, enabling motion background modeling. Last, we propose a multiframe joint query decoder. The shallowest feature map after multiscale feature propagation is mapped to initial query weights, which are refined through grouped convolutions to generate grouped query vectors. These are jointly optimized to encapsulate rich multiframe details, strengthening motion background modeling and target feature representation, improving prediction accuracy. Experimental results on the NUDT-MIRSDT, IRDST, and the established TSIRMT datasets demonstrate that our network outperforms state-of-the-art (SOTA) methods.
</p>



# Architecture
<p align="center">
  <img src="pic/LMAFormer_fig_1.png" width="auto" alt="accessibility text">
</p>
Overall Architecture of LMAFormer.

# Installation

## Linux environment

Python 3.10 and an NVIDIA GPU with CUDA are recommended. The custom
MultiScaleDeformableAttention extension must be compiled on the same Linux
machine and against the same CUDA version as PyTorch.

```bash
conda create -n LMAFormer python=3.10 -y
conda activate LMAFormer

# Select the command matching the machine's driver from
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -r requirements.txt
```

Install a CUDA toolkit that provides `nvcc` and matches `torch.version.cuda`.
For example, when the PyTorch environment uses CUDA 12.1:

```bash
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

Then build the custom CUDA extension from this repository:

```bash
cd MFIRSTD/models/ops
python setup.py build install
cd ../../..
python -c "import MultiScaleDeformableAttention"
```

If the machine has no CUDA-capable GPU, the project can only be prepared for
development; training and inference require a CUDA device.

## Datasets
We evaluate network performance using NUDR-MIRSDT, IRDST and a self-built dataset TSIRMT

Download the datasets following the corresponding paper/project page. Set
`LMAFORMER_DATASET_ROOT` to the directory containing `ImageSets`, `images`, and
`masks`; no source-code edit is needed:

```bash
export LMAFORMER_DATASET_ROOT=/data/LMAFormer/IRDST
```

For the TeiKe helper scripts, point `LMAFORMER_TEIKE_ROOT` at the directory
containing `data/`:

```bash
export LMAFORMER_TEIKE_ROOT=/data/LMAFormer/TeiKe
```

Here is the list of datasets used. 

- [NUDT-MIRSDT](https://pan.baidu.com/s/1pSN350eurMafLiHBQBnrPA?pwd=5whn) (Extraction code: 5whn)
- [IRDST](https://drive.google.com/file/d/1sb-32pydlpXvlNxwx9niT2t6KP9oMJID/view?usp=sharing)
- [TSIRMT](https://drive.google.com/drive/folders/1aWDNdUWkTOuV3fILbgLDEqM2N2erW05n?usp=sharing)

The NUDT-MIRSDT dataset requires the 'rearrange_dataset.py' program for path reconstruction, with 'rearrange_dataset.py' located at '/Datasets/rearrange_dataset.py'.
The dataset division is subject to the method in the Datasets folder.
## Download Trained Models 
Pretrained Swin backbones can be downloaded from their corresponding repository. 
- [Swin-S](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth)
- [Swin-B](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth)
- [Resnet-101 COCO](https://drive.google.com/file/d/1NGuaew1d0x0kdK2XR_J3Vlmc6OGtOe58/view?usp=sharing)

If you are interested in evaluating only, you can download the selected trained LMAFormer checkpoints from the links in the results table.   


# Training 

The models were trained and tested using a single NVIDIA 4080 GPU.  

* Train LMAFormer with Swin backbone on NUDT-MIRSDR, IRDST, TSIRMT datasets:
  ```
  python train_swin.py
  ```

# Inference

### Ablation study on MAFEM (CME / EnE)
See [ABLATION.md](ABLATION.md). The four settings are enabled with
`--ablation full | no_mafem | mafem_no_cme | mafem_no_ene`, and
`python run_ablation.py` trains, evaluates and tabulates all of them.

### Inference on NUDT-MIRSDT:

```bash
python inference_swin.py --model_path ./result/NUDT-MIRSDT/checkpoint_NUDT-MIRSDT.pth --dataset NUDT-MIRSDT --val_size 400 --flip --msc --output_dir ./predict/NUDT-MIRSDT
```

Expected mIoU: 73.26

### Inference on IRDST:

```bash
python inference_swin.py --model_path ./result/IRDST/checkpoint_IRDST.pth --dataset IRDST --val_size 400 --flip --msc --output_dir ./predict/IRDST
```

Expected mIoU: 59.17

### Inference on TSIRMT:

```bash
python inference_swin.py --model_path ./result/TSIRMT/checkpoint_TSIRMT.pth --dataset TSIRMT --val_size 400 --flip --msc --output_dir ./predict/TSIRMT
```

Expected mIoU: 65.89

## Results Summary
### Results on NUDT-MIRSDT, IRDST and TSIRMT
| Dataset  | Checkpoint                                                                                        | IoU  | nIoU | Pd | Fa |
|-----------|---------------------------------------------------------------------------------------------------|------|------|------|------|
| NUDT-MIRSDT | [checkpoint](https://drive.google.com/file/d/1jBYw9jWAhTf7_pCRNWLfBMflhMq25_Vn/view?usp=sharing)  | 73.26 | 73.63 | 99.68 | 0.71 |
| IRDST | [checkpoint](https://drive.google.com/file/d/1LY2OWb3FyXB568vHpakJog1_qaDKOYZO/view?usp=sharing)  | 59.17 | 57.51 | 99.64 | 14.95 |
| TSIRMT | [checkpoint](https://drive.google.com/file/d/1lnVRmUSnrqZIF_j50z5hovpqiaNB4STY/view?usp=sharing)  | 65.89 | 65.63 | 86.10 | 185.78 |

### Acknowledgement
We would like to thank the open-source projects with  special thanks to [DETR](https://github.com/facebookresearch/detr)  and [VisTR](https://github.com/Epiphqny/VisTR) for making their code public. Part of the code in our project are collected and modified from several open source repositories.

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@ARTICLE{10758760,
  author={Huang, Yuanxin and Zhi, Xiyang and Hu, Jianming and Yu, Lijian and Han, Qichao and Chen, Wenbin and Zhang, Wei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={LMAFormer: Local Motion Aware Transformer for Small Moving Infrared Target Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-17},
  keywords={Feature extraction;Object detection;Transformers;Decoding;Three-dimensional displays;Computational modeling;Deep learning;Annotations;Visualization;Urban areas;Infrared small moving target detection;local motion aware;multiframe joint query;multiscale transformer encoder},
  doi={10.1109/TGRS.2024.3502663}}
```

