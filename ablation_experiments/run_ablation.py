"""
Run ablation experiments and save comparison data and plots.

Configuration is via editing the constants below or environment variables prefixed with
`AB_` (see defaults). The script will write results into `Kaggle/ablation_experiments/results/`.

Defaults: uses `kaggle_unet/img/train_img` and `kaggle_unet/img/train_mask` as data, and
`Kaggle/weights/best.pth` as the model weight for experiments where a weight is needed.

This script by default evaluates only the first `SAMPLE_N` validation images (default 200)
to run quickly; set `AB_SAMPLE_N` env var to `0` to use all images.
"""

import os
from glob import glob
from pathlib import Path
import csv
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import importlib.util
from pathlib import Path as _Path

# load plot_of1_vs_threshold.py as module (Kaggle may not be an importable package)
_this_dir = _Path(__file__).resolve().parent
_kaggle_dir = _this_dir.parent
_plot_path = _kaggle_dir / 'plot_of1_vs_threshold.py'
spec = importlib.util.spec_from_file_location('plot_of1_vs_threshold', str(_plot_path))
plotmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotmod)

# ----------------- Configuration (edit or use env vars) -----------------
WEIGHT_DEFAULT = os.environ.get('AB_WEIGHTS', 'Kaggle/weights/best.pth')
IMAGES_DIR = os.environ.get('AB_IMAGES_DIR', 'kaggle_unet/img/train_img')
MASKS_DIR = os.environ.get('AB_MASKS_DIR', 'kaggle_unet/img/train_mask')
OUT_DIR = os.environ.get('AB_OUT_DIR', 'Kaggle/ablation_experiments/results')
DEVICE = os.environ.get('AB_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_N = int(os.environ.get('AB_SAMPLE_N', '200'))
IMG_SIZE = int(os.environ.get('AB_IMG_SIZE', '512'))

# Experiments definitions: name -> dict(weights, hole_fill, note)
EXPERIMENTS = {
    'baseline_no_fill': {'weights': WEIGHT_DEFAULT, 'hole_fill': False, 'note': 'Baseline without hole filling'},
    'with_hole_fill': {'weights': WEIGHT_DEFAULT, 'hole_fill': True, 'note': 'Enable hole filling postprocessing'},
    # for BCE-only you must supply a weight trained with BCE-only; if not present this entry will be skipped
    'bce_only': {'weights': os.environ.get('AB_BCE_WEIGHTS', ''), 'hole_fill': False, 'note': 'Model trained with BCE only (provide weight path in AB_BCE_WEIGHTS)'},
}

# thresholds for threshold sweep (used for plots)
THRESHOLDS = np.arange(0.0, 1.0 + 1e-9, 0.01)

# specific thresholds comparison for fixed-weight test
FIXED_THRESHOLDS = [0.3, 0.5]

# -------------------------------------------------------------------------


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def postprocess_mask(bin_mask, hole_fill=False):
    # bin_mask: HxW uint8
    if not hole_fill:
        return bin_mask
    # morphological closing then fill small holes via floodFill
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(bin_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)

    # fill holes using floodFill on inverted mask
    h, w = closed.shape
    mask = closed.copy()
    # flood fill background
    im_floodfill = mask.copy()
    mask_8u = (im_floodfill * 255).astype('uint8')
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_8u, ff_mask, (0, 0), 255)
    # invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(mask_8u)
    filled = cv2.bitwise_or((closed * 255).astype('uint8'), im_floodfill_inv)
    return (filled > 0).astype(np.uint8)


def evaluate_experiment(name, cfg, val_imgs):
    weights = cfg.get('weights')
    hole_fill = cfg.get('hole_fill', False)
    if not weights or not os.path.exists(weights):
        print(f"Skipping {name}: weight not found ({weights})")
        return None

    print(f"Running {name} with weights={weights} hole_fill={hole_fill}")
    # determine backbone: prefer AB_BACKBONE env, else fallback to train.py constant if available, else 'resnet34'
    backbone = os.environ.get('AB_BACKBONE')
    try:
        if not backbone and hasattr(plotmod, 'train_module') and plotmod.train_module is not None:
            backbone = plotmod.train_module.BACKBONE
    except Exception:
        backbone = backbone
    backbone = backbone or 'resnet34'
    model = plotmod.load_model(weights, DEVICE, backbone=backbone)

    tp_total = {t: 0 for t in THRESHOLDS}
    fp_total = {t: 0 for t in THRESHOLDS}
    fn_total = {t: 0 for t in THRESHOLDS}

    for img_path in tqdm(val_imgs, desc=f'Eval {name}'):
        base = Path(img_path).stem
        mask_path = Path(MASKS_DIR) / f"{base}.npy"
        if mask_path.exists():
            gt = np.load(mask_path)
            if gt.ndim == 3:
                gt = (gt.sum(axis=0) > 0).astype(np.uint8)
            else:
                gt = (gt > 0).astype(np.uint8)
            gt = cv2.resize(gt.astype('uint8'), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        else:
            gt = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # preprocess image same as plotmod
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        inp = torch.from_numpy(img).unsqueeze(0).to(DEVICE).float()

        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        for t in THRESHOLDS:
            bin_pred = (probs > t).astype(np.uint8)
            bin_pred = postprocess_mask(bin_pred, hole_fill=hole_fill)
            pred_objs = plotmod.connected_components_masks(bin_pred)
            gt_objs = plotmod.connected_components_masks(gt)
            tp, fp, fn = plotmod.match_objects(pred_objs, gt_objs, min_iou=float(os.environ.get('AB_MATCH_IOU', 0.5)))
            tp_total[t] += tp
            fp_total[t] += fp
            fn_total[t] += fn

    of1s = {t: (2 * tp_total[t]) / (2 * tp_total[t] + fp_total[t] + fn_total[t]) if (2 * tp_total[t] + fp_total[t] + fn_total[t]) > 0 else 1.0 for t in THRESHOLDS}

    # save csv
    csv_path = Path(OUT_DIR) / f"{name}_of1.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold', 'oF1', 'TP', 'FP', 'FN'])
        for t in THRESHOLDS:
            writer.writerow([t, of1s[t], tp_total[t], fp_total[t], fn_total[t]])

    # plot
    plt.figure(figsize=(8, 4))
    plt.plot(list(THRESHOLDS), [of1s[t] for t in THRESHOLDS], marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('oF1')
    plt.title(f'{name} oF1 vs Threshold')
    plt.grid(True)
    plt.tight_layout()
    plot_path = Path(OUT_DIR) / f"{name}_of1.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # fixed thresholds comparison (0.3 & 0.5)
    fixed_results = {}
    for t in FIXED_THRESHOLDS:
        fixed_results[t] = {
            'oF1': of1s[t],
            'TP': tp_total[t],
            'FP': fp_total[t],
            'FN': fn_total[t]
        }

    return {'name': name, 'csv': str(csv_path), 'plot': str(plot_path), 'fixed': fixed_results, 'note': cfg.get('note', '')}


def write_markdown(results_list):
    md_path = Path(OUT_DIR) / 'ablation_report.md'
    with open(md_path, 'w') as f:
        f.write('# Ablation Experiments Report\n\n')
        f.write('## Summary\n\n')
        for r in results_list:
            if r is None:
                continue
            f.write(f"- **{r['name']}**: {r.get('note','')}\n")
        f.write('\n')

        f.write('## Fixed-threshold comparison (0.3 vs 0.5)\n\n')
        f.write('|Experiment|Thresh|oF1|TP|FP|FN|\n')
        f.write('|-|-|-|-|-|-|\n')
        for r in results_list:
            if r is None:
                continue
            for t, vals in r['fixed'].items():
                f.write(f"|{r['name']}|{t}|{vals['oF1']:.4f}|{vals['TP']}|{vals['FP']}|{vals['FN']}|\n")

        f.write('\n## Plots and CSVs\n\n')
        for r in results_list:
            if r is None:
                continue
            f.write(f"### {r['name']}\n")
            f.write(f"![plot]({Path(r['plot']).name})\n\n")
            f.write(f"CSV: {Path(r['csv']).name}\n\n")

    print(f'Wrote report to {md_path}')


def main():
    ensure_out()
    img_files = sorted(glob(os.path.join(IMAGES_DIR, '*.png'))) + sorted(glob(os.path.join(IMAGES_DIR, '*.jpg'))) + sorted(glob(os.path.join(IMAGES_DIR, '*.jpeg')))
    if len(img_files) == 0:
        raise RuntimeError(f'No images found in {IMAGES_DIR}')
    # use last VAL_SPLIT portion as validation to be consistent with train.py
    # here we simply use the tail half logic based on env var AB_VAL_SPLIT or default 0.2
    val_split = float(os.environ.get('AB_VAL_SPLIT', '0.2'))
    n_total = len(img_files)
    n_train = int((1 - val_split) * n_total)
    val_imgs = img_files[n_train:]

    if SAMPLE_N > 0:
        val_imgs = val_imgs[:SAMPLE_N]

    results = []
    for name, cfg in EXPERIMENTS.items():
        res = evaluate_experiment(name, cfg, val_imgs)
        results.append(res)

    write_markdown(results)


if __name__ == '__main__':
    main()
