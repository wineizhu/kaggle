"""
Copy-Move Forgery Detection Inference Script
使用训练好的smp.Unet模型进行推理
"""

import os
import sys
from pathlib import Path
from glob import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("ERROR: segmentation_models_pytorch not installed")
    sys.exit(1)

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==================== 配置参数 ====================
WEIGHTS_PATH = "./weights/checkpoint_epoch20.pth"
TEST_IMG_DIR = "./img/test_img"
OUTPUT_DIR = "./result"

IMG_SIZE = 512
BATCH_SIZE = 1
USE_CUDA = True
DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

BACKBONE = "resnet34"
THRESH = 0.3
TTA = False  # Test-time augmentation

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 推理函数 ====================
def get_inference_transforms(size=IMG_SIZE):
    """Get inference transforms"""
    transforms = A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    return transforms


def load_model(weights_path, device=DEVICE):
    """Load trained model"""
    print(f"Loading model from {weights_path}...")
    
    model = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Model loaded successfully")
    else:
        print(f"ERROR: Weights file not found: {weights_path}")
        sys.exit(1)
    
    return model


def infer_image(model, img_path, transforms, device, thresh=0.5, tta=False):
    """Run inference on single image"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    # 应用变换
    transformed = transforms(image=img)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
    
    # 二值化
    mask = (probs > thresh).astype(np.uint8)
    
    # 调整回原始大小
    if mask.shape != (orig_h, orig_w):
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    return mask


def post_process(mask, min_area=50):
    """Post-process mask"""
    # 移除小连通分量
    num_labels, labels = cv2.connectedComponents(mask)
    processed = np.zeros_like(mask)
    
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        area = component.sum()
        if area >= min_area:
            processed[component > 0] = 1
    
    return processed


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("Copy-Move Forgery Detection Inference")
    print("=" * 70)
    
    # 加载模型
    print(f"\n[1/3] Loading model...")
    model = load_model(WEIGHTS_PATH, DEVICE)
    print(f"Device: {DEVICE}")
    
    # 获取测试图像
    print(f"\n[2/3] Preparing test images from {TEST_IMG_DIR}...")
    
    if not os.path.exists(TEST_IMG_DIR):
        print(f"ERROR: Test image directory not found: {TEST_IMG_DIR}")
        sys.exit(1)
    
    test_imgs = sorted(glob(os.path.join(TEST_IMG_DIR, "*.png"))) + \
                sorted(glob(os.path.join(TEST_IMG_DIR, "*.jpg"))) + \
                sorted(glob(os.path.join(TEST_IMG_DIR, "*.jpeg")))
    
    print(f"Found {len(test_imgs)} test images")
    
    # 获取变换
    transforms = get_inference_transforms(IMG_SIZE)
    
    # 推理
    print(f"\n[3/3] Running inference...")
    
    model.eval()
    for i, img_path in enumerate(tqdm(test_imgs), 1):
        base_name = Path(img_path).stem
        
        try:
            # 推理
            mask = infer_image(model, img_path, transforms, DEVICE, THRESH, TTA)
            
            # 后处理
            mask = post_process(mask, min_area=50)
            
            # 保存.npy格式
            npy_path = os.path.join(OUTPUT_DIR, f"{base_name}.npy")
            mask_3d = np.expand_dims(mask, axis=0).astype(np.uint8)
            np.save(npy_path, mask_3d)
            
            # 保存可视化PNG
            png_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
            png_mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(png_path, png_mask)

            # 生成并保存原图叠加效果图（overlay）
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                # 确保掩码与原图尺寸一致
                if mask.shape[:2] != img_bgr.shape[:2]:
                    mask_resized = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask

                # 颜色叠加：红色区域标注篡改，alpha 控制透明度
                color = (0, 0, 255)  # BGR 红色
                alpha = 0.45
                color_layer = np.zeros_like(img_bgr)
                color_layer[mask_resized > 0] = color
                overlay_img = cv2.addWeighted(img_bgr, 1.0, color_layer, alpha, 0)

                # 可选：描边轮廓提升可见性（黄色）
                contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_img, contours, -1, (0, 255, 255), thickness=2)

                overlay_path = os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png")
                cv2.imwrite(overlay_path, overlay_img)
            
        except Exception as e:
            print(f"ERROR processing {base_name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Inference complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
