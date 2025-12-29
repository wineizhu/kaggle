"""
Copy-Move Forgery Detection Training Script
使用 segmentation_models_pytorch (smp) 的高性能UNet模型
"""

import os
import sys
import random
from glob import glob
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
    print(f"✓ Using segmentation_models_pytorch")
except ImportError:
    print("ERROR: segmentation_models_pytorch not installed")
    print("Install with: pip install segmentation-models-pytorch")
    sys.exit(1)

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==================== 配置参数 ====================
IMAGES_DIR = "./img/train_img"
MASKS_DIR = "./img/train_mask"
SAVE_DIR = "./weights"

IMG_SIZE = 512
BATCH_SIZE = 30
EPOCHS = 50
LR = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 0  # Set to 0 for faster loading
SEED = 42
PATIENCE = 10  # Early stopping

BACKBONE = "resnet34"
USE_PRETRAINED = True
USE_CUDA = True
DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)


# ==================== 工具函数 ====================
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# set_seed(SEED)


# ==================== 数据加载 ====================
class ForgeryDataset(Dataset):
    """Copy-move forgery detection dataset"""
    
    def __init__(self, image_paths, masks_dir, transforms=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = img_path.stem

        # 读取图像
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 读取掩码
        mask_path = self.masks_dir / f"{base}.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
            # 处理多通道掩码 (C, H, W)
            if mask.ndim == 3:
                mask = (mask.sum(axis=0) > 0).astype(np.uint8)
            else:
                mask = (mask > 0).astype(np.uint8)
            # 调整掩码大小
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # 应用数据增强
        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img = data['image']
            mask = data['mask'].unsqueeze(0).float()  # (1, H, W)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask


def get_transforms(size=IMG_SIZE):
    """Get data augmentation transforms"""
    train_transforms = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], keypoint_params=None)

    val_transforms = A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], keypoint_params=None)

    return train_transforms, val_transforms


# ==================== 损失函数 ====================
def get_loss_fn():
    """Get combined loss function"""
    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary")
    
    def loss_fn(pred, gt):
        return 0.5 * bce(pred, gt) + 0.5 * dice(pred, gt)
    
    return loss_fn


# ==================== 指标计算 ====================
def compute_dice(pred, target, thresh=0.5):
    """计算Dice系数"""
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred) > thresh).float()
        target_binary = (target > 0).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        if union == 0:
            return 1.0
        
        dice = 2.0 * intersection / (union + 1e-6)
        return dice.item()


def compute_iou(pred, target, thresh=0.5):
    """计算IoU"""
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred) > thresh).float()
        target_binary = (target > 0).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = (pred_binary + target_binary - pred_binary * target_binary).sum()
        
        if union == 0:
            return 1.0
        
        iou = intersection / (union + 1e-6)
        return iou.item()


# ==================== 训练和验证 ====================
def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for img, mask in pbar:
        img = img.to(device)
        mask = mask.to(device)
        
        # Forward pass
        logits = model(img)
        loss = loss_fn(logits, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_dice += compute_dice(logits, mask)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    
    return avg_loss, avg_dice


def validate(model, val_loader, loss_fn, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc="Validating"):
            img = img.to(device)
            mask = mask.to(device)
            
            logits = model(img)
            loss = loss_fn(logits, mask)
            
            total_loss += loss.item()
            total_dice += compute_dice(logits, mask)
            total_iou += compute_iou(logits, mask)
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    
    return avg_loss, avg_dice, avg_iou


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("Copy-Move Forgery Detection Training")
    print("=" * 70)
    
    # 数据准备
    print(f"\n[1/5] Preparing data from {IMAGES_DIR}...")
    
    if not os.path.exists(IMAGES_DIR):
        print(f"ERROR: Image directory not found: {IMAGES_DIR}")
        sys.exit(1)
    
    if not os.path.exists(MASKS_DIR):
        print(f"ERROR: Mask directory not found: {MASKS_DIR}")
        sys.exit(1)
    
    # 获取图像列表
    img_files = sorted(glob(os.path.join(IMAGES_DIR, "*.png"))) + \
                sorted(glob(os.path.join(IMAGES_DIR, "*.jpg"))) + \
                sorted(glob(os.path.join(IMAGES_DIR, "*.jpeg")))
    
    print(f"Found {len(img_files)} images")
    
    # 验证集分割
    n_total = len(img_files)
    n_train = int((1 - VAL_SPLIT) * n_total)
    
    train_imgs = img_files[:n_train]
    val_imgs = img_files[n_train:]
    
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    
    # 创建数据加载器
    print("\n[2/5] Creating data loaders...")
    train_transforms, val_transforms = get_transforms(IMG_SIZE)
    
    train_ds = ForgeryDataset(train_imgs, MASKS_DIR, train_transforms)
    val_ds = ForgeryDataset(val_imgs, MASKS_DIR, val_transforms)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 模型创建
    print(f"\n[3/5] Creating model (backbone={BACKBONE})...")
    model = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights="imagenet" if USE_PRETRAINED else None,
        in_channels=3,
        classes=1
    ).to(DEVICE)
    

    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    print("\n[4/5] Setting up optimizer and loss...")
    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = get_loss_fn()
    
    # 训练循环
    print(f"\n[5/5] Starting training for {EPOCHS} epochs...\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 记录指标
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': []
    }
    # print("model\n", model)
    # a = int(input())
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*70}")
        
        # 训练
        train_loss, train_dice = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        
        # 验证
        val_loss, val_dice, val_iou = validate(model, val_loader, loss_fn, DEVICE)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        # 打印结果
        print(f"\nTrain Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_path = os.path.join(SAVE_DIR, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved best model to {best_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # 保存检查点
        if epoch % 10 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ Saved checkpoint to {ckpt_path}")
    
    # 打印最终结果
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Model saved to: {os.path.join(SAVE_DIR, 'best.pth')}")
    print()


if __name__ == "__main__":
    main()
