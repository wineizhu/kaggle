"""
Compare predictions from ablation experiments on a single validation image.

Configuration via environment variables:
- AB_IMAGE_PATH: relative path to image to use (defaults to first val image)
- AB_OUT_DIR: output directory (defaults to Kaggle/ablation_experiments/results)
- AB_WEIGHTS / AB_BCE_WEIGHTS: as used by run_ablation
"""

import os
from glob import glob
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import importlib.util
from pathlib import Path as _Path

# load plot_of1_vs_threshold as plotmod
_this_dir = _Path(__file__).resolve().parent
_kaggle_dir = _this_dir.parent
_plot_path = _kaggle_dir / 'plot_of1_vs_threshold.py'
spec = importlib.util.spec_from_file_location('plot_of1_vs_threshold', str(_plot_path))
plotmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotmod)

# config
OUT_DIR = Path(os.environ.get('AB_OUT_DIR', str(_this_dir / 'results')))
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_PATH = os.environ.get('AB_IMAGE_PATH')
IMAGES_DIR = os.environ.get('AB_IMAGES_DIR', 'kaggle_unet/img/train_img')
MASKS_DIR = os.environ.get('AB_MASKS_DIR', 'kaggle_unet/img/train_mask')
WEIGHTS = os.environ.get('AB_WEIGHTS', 'Kaggle/weights/best.pth')
BCE_WEIGHTS = os.environ.get('AB_BCE_WEIGHTS', WEIGHTS)
DEVICE = os.environ.get('AB_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = int(os.environ.get('AB_IMG_SIZE', '512'))

EXPS = [
    ('baseline_no_fill', WEIGHTS, False),
    ('with_hole_fill', WEIGHTS, True),
    ('bce_only', BCE_WEIGHTS, False),
]

THRESHOLDS = [0.3, 0.5]


def load_image_and_gt(img_path):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return img_resized, gt, base


def preprocess_for_model(img_rgb):
    img = img_rgb.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return img


def overlay_mask(img_rgb, mask, color=(255, 0, 0), alpha=0.4):
    # img_rgb HxWx3, mask HxW (0/1)
    over = img_rgb.copy().astype(np.float32)
    col = np.array(color, dtype=np.float32)
    over[mask > 0] = over[mask > 0] * (1 - alpha) + col * alpha
    return over.astype(np.uint8)


def run():
    # pick image
    if IMAGE_PATH:
        img_path = Path(IMAGE_PATH)
    else:
        imgs = sorted(glob(os.path.join(IMAGES_DIR, '*.png'))) + sorted(glob(os.path.join(IMAGES_DIR, '*.jpg')))
        if len(imgs) == 0:
            raise RuntimeError(f'No images found in {IMAGES_DIR}')
        # use first validation image similar to run_ablation logic (use tail portion)
        val_split = float(os.environ.get('AB_VAL_SPLIT', '0.2'))
        n_total = len(imgs)
        n_train = int((1 - val_split) * n_total)
        val_imgs = imgs[n_train:]
        img_path = Path(val_imgs[0])

    img_rgb, gt, base = load_image_and_gt(img_path)

    # whether to include Original and GT (set AB_INCLUDE_ORIG_GT=0 to hide)
    include_orig_gt = os.environ.get('AB_INCLUDE_ORIG_GT', '1') not in ('0', 'false', 'False')

    # Fixed 2x3 layout (2 rows x 3 columns) to display the 3 experiments x 2 thresholds
    nrows = 2
    ncols = 3

    # build list of panels (title, image) in the order of EXPS x THRESHOLDS
    panels = []
    for exp_name, weight_path, hole_fill in EXPS:
        if not weight_path or not Path(weight_path).exists():
            for t in THRESHOLDS:
                panels.append((f"{exp_name}\n(skipped)", None))
            continue

        model = plotmod.load_model(weight_path, DEVICE, backbone=os.environ.get('AB_BACKBONE', None) or 'resnet34')
        inp = preprocess_for_model(img_rgb)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            logits = model(inp_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        for t in THRESHOLDS:
            bin_pred = (probs > t).astype(np.uint8)
            if hole_fill:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                bin_pred = cv2.morphologyEx(bin_pred.astype('uint8'), cv2.MORPH_CLOSE, kernel)
                h, w = bin_pred.shape
                mask_8u = (bin_pred * 255).astype('uint8')
                ff_mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(mask_8u, ff_mask, (0, 0), 255)
                im_floodfill_inv = cv2.bitwise_not(mask_8u)
                filled = cv2.bitwise_or((bin_pred * 255).astype('uint8'), im_floodfill_inv)
                bin_pred = (filled > 0).astype(np.uint8)

            over = overlay_mask(img_rgb, bin_pred, color=(255, 0, 0), alpha=0.4)
            panels.append((f"{exp_name}\nth={t}", over))

    # create figure and place panels row-major into 2x3 grid
    fig = plt.figure(figsize=(ncols * 3.0, nrows * 3.0))
    gs = plt.GridSpec(nrows, ncols, figure=fig, wspace=0.12, hspace=0.22)

    for idx in range(nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])
        if idx < len(panels):
            title, img_panel = panels[idx]
            if img_panel is None:
                ax.text(0.5, 0.5, title, ha='center', va='center')
            else:
                ax.imshow(img_panel)
            ax.set_title(title)
        else:
            ax.axis('off')
        ax.axis('off')

    # if user requested Original/GT, save them as a separate small image
    if include_orig_gt:
        fig2 = plt.figure(figsize=(6, 3))
        ax = fig2.add_subplot(1, 2, 1)
        ax.imshow(img_rgb)
        ax.set_title('Original')
        ax.axis('off')
        ax = fig2.add_subplot(1, 2, 2)
        ax.imshow(gt, cmap='gray')
        ax.set_title('GT')
        ax.axis('off')
        out_orig = OUT_DIR / f"compare_origgt_{base}.png"
        plt.tight_layout()
        plt.savefig(out_orig, dpi=150)
        plt.close(fig2)

    suffix = '' if include_orig_gt else '_noorig'
    out_path = OUT_DIR / f"compare{suffix}_{base}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison to {out_path}")


if __name__ == '__main__':
    run()
