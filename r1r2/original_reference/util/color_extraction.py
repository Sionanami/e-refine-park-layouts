# 提取颜色相关（extract_color_regions, extract_color_mask等）
import numpy as np

def extract_color_regions(image, target_rgb, tolerance=50):
    img_np = np.array(image)
    dist = np.linalg.norm(img_np - np.array(target_rgb), axis=2)
    return (dist < tolerance).astype(np.uint8)

def extract_color_mask(image, target_rgb, tolerance=50):
    img_np = np.array(image)
    dist = np.linalg.norm(img_np - np.array(target_rgb), axis=2)
    return (dist < tolerance).astype(np.uint8)


import numpy as np
from PIL import Image

import numpy as np

def extract_color_regions(image_pil, target_rgb, tolerance=30):
    """
    输入：PIL图像，目标颜色tuple，容差
    输出：bool掩码 numpy数组
    """
    image_np = np.array(image_pil)
    lower = np.array([max(0, c - tolerance) for c in target_rgb])
    upper = np.array([min(255, c + tolerance) for c in target_rgb])

    mask = np.all((image_np >= lower) & (image_np <= upper), axis=2)
    return mask

def extract_orange_region(image_pil, orange_rgb, tolerance=30):
    """
    专门提取橙色，内部可做特殊处理（如果有）
    """
    return extract_color_regions(image_pil, orange_rgb, tolerance)
