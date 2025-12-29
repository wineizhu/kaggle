"""
Generate overlay images for training and test sets
叠加掩码到原图上，用于可视化标注区域
"""
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path


def generate_overlay(img_path, mask_path, output_path, color=(0, 0, 255), alpha=0.45, draw_contours=True):
    """
    生成单个overlay图片
    
    Args:
        img_path: 原图路径
        mask_path: 掩码路径 (.npy格式)
        output_path: 输出路径
        color: BGR格式颜色，默认红色
        alpha: 透明度，0-1之间
        draw_contours: 是否绘制轮廓
    """
    # 读取原图
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠ Cannot read image: {img_path}")
        return False
    
    # 读取掩码
    if not os.path.exists(mask_path):
        print(f"⚠ Mask not found: {mask_path}")
        return False
    
    mask = np.load(mask_path)
    
    # 处理多通道掩码 (C, H, W) -> (H, W)
    if mask.ndim == 3:
        mask = mask.max(axis=0)  # 取最大值合并通道
    
    # 二值化
    mask = (mask > 0).astype(np.uint8)
    
    # 调整掩码尺寸与原图一致
    if mask.shape[:2] != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 创建彩色图层
    color_layer = np.zeros_like(img_bgr)
    color_layer[mask > 0] = color
    
    # 混合原图与彩色图层
    overlay_img = cv2.addWeighted(img_bgr, 1.0, color_layer, alpha, 0)
    
    # 绘制轮廓（可选）
    if draw_contours:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, contours, -1, (0, 255, 255), thickness=2)  # 黄色轮廓
    
    # 保存
    cv2.imwrite(output_path, overlay_img)
    return True


def process_dataset(img_dir, mask_dir, output_dir, dataset_name="dataset"):
    """
    批量处理数据集
    
    Args:
        img_dir: 图片目录
        mask_dir: 掩码目录
        output_dir: 输出目录
        dataset_name: 数据集名称（用于显示）
    """
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name}")
    print(f"{'='*70}")
    
    # 检查目录是否存在
    if not os.path.exists(img_dir):
        print(f"⚠ Image directory not found: {img_dir}")
        return
    
    if not os.path.exists(mask_dir):
        print(f"⚠ Mask directory not found: {mask_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片
    img_patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    img_files = []
    for pattern in img_patterns:
        img_files.extend(glob(os.path.join(img_dir, pattern)))
    
    img_files = sorted(img_files)
    print(f"Found {len(img_files)} images in {img_dir}")
    
    # 处理每张图片
    success_count = 0
    fail_count = 0
    
    for img_path in tqdm(img_files, desc=f"Generating overlays"):
        # 获取基础文件名（不含扩展名）
        base_name = Path(img_path).stem
        
        # 构造对应的掩码路径
        mask_path = os.path.join(mask_dir, f"{base_name}.npy")
        
        # 构造输出路径
        output_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        
        # 生成overlay
        if generate_overlay(img_path, mask_path, output_path):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n✓ {dataset_name}: {success_count} overlays generated, {fail_count} failed")
    print(f"  Saved to: {output_dir}")


def main():
    """主函数"""
    print("="*70)
    print("Overlay Image Generator")
    print("="*70)
    
    # 定义路径
    base_dir = "./img"
    
    # 训练集
    train_img_dir = os.path.join(base_dir, "train_img")
    train_mask_dir = os.path.join(base_dir, "train_mask")
    train_output_dir = os.path.join(base_dir, "train_overlay")
    
    # 测试集
    test_img_dir = os.path.join(base_dir, "test_img")
    test_mask_dir = os.path.join(base_dir, "test_mask")
    test_output_dir = os.path.join(base_dir, "test_overlay")
    
    # 处理训练集
    process_dataset(train_img_dir, train_mask_dir, train_output_dir, "Training Set")
    
    # 处理测试集
    process_dataset(test_img_dir, test_mask_dir, test_output_dir, "Test Set")
    
    print(f"\n{'='*70}")
    print("✓ All overlays generated successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
