# HUGS-Net

Clean training project for the complete **HUe-Guided Synergistic Network.

## Model mapping

The complete model
- Its parameter count was 1.722982M before cleanup, matching the reported 1.72M.

The implementation is now exposed as `models.HUGSNet`. Module names follow
Section IV of the paper:

- `HGBranch` (Hue Guidance Branch)
- `DDRBranch` (Downscale Detail Refinement Branch)
- `MultiScaleFusionStage` (cross-branch scale transfer and reconstruction)
- `FourierEmbeddedBlock` (FEB)
- `ResidueGroupConvolutionBlock` (RGCB)
- `MultiScaleFusionBlock` (MFB)

## Dataset layout

```text
dataset_root/
  train/
    input/
    gt/
  valid/
    input/
    gt/
  test/              # optional; pass --split test
    input/
    gt/
```

Files in `input` and `gt` must have matching names. Training crops paired
128x128 patches by default, as described in the paper.

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
python train.py --dataset-dir /path/to/dataset_root
```

Resume training:

```bash
python train.py \
  --dataset-dir /path/to/dataset_root \
  --resume results/hugs_net/checkpoints/last.pt
```

## Test

```bash
python test.py \
  --dataset-dir /path/to/dataset_root \
  --split valid \
  --checkpoint results/hugs_net/checkpoints/best.pt \
  --save-images
```

## Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{11175439,
  author={Zhang, Ting and Wang, Runjie and Niu, Yuzhen and Li, Zuoyong and Zhao, Tiesong},
  journal={IEEE Transactions on Multimedia},
  title={HUGS-Net: A Lightweight and Unified Network for Adverse Weather Image Denoising},
  year={2025},
  volume={27},
  number={},
  pages={9570-9580},
  keywords={Meteorology;Noise;Feature extraction;Image color analysis;Image restoration;Data mining;Computational modeling;Training;Rain;Noise measurement;Image denoising;adverse weather condition;deraining;dehazing;desnowing;color space transformation},
  doi={10.1109/TMM.2025.3613104}
}
```
