# Ablation Experiments

This folder contains scripts to run the requested ablation experiments and generate a markdown report with CSVs and plots.

How to run (recommended inside the `deeplearning_zyh` conda env):

```bash
# from repo root
conda run -n deeplearning_zyh python Kaggle/ablation_experiments/run_ablation.py
```

Configuration:
- Edit `AB_` environment variables or the top of `run_ablation.py` to set paths and options.

Useful env vars:
- `AB_WEIGHTS` default `Kaggle/weights/best.pth`
- `AB_BCE_WEIGHTS` path to model trained with BCE-only (for the `bce_only` experiment)
- `AB_IMAGES_DIR` default `kaggle_unet/img/train_img`
- `AB_MASKS_DIR` default `kaggle_unet/img/train_mask`
- `AB_SAMPLE_N` default `200` (set to `0` to use all validation images)
- `AB_OUT_DIR` output folder (default `Kaggle/ablation_experiments/results`)

Outputs:
- `results/ablation_report.md` — markdown summary with tables and embedded plots
- Per-experiment CSVs and PNG plots in `results/`.
