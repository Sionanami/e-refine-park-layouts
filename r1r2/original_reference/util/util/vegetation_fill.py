# 植物填充逻辑

import cv2
import numpy as np
from PIL import Image, ImageDraw

# 颜色提取函数
def extract_color_mask(image, target_rgb, tolerance=50):
    img_np = np.array(image)
    dist = np.linalg.norm(img_np - np.array(target_rgb), axis=2)
    mask = dist < tolerance
    return mask.astype(np.uint8)

# 计算橙色最短边宽
def get_short_edge_width_of_orange(image, orange_mask, default_radius=5):
    mask_cv = (orange_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_edge = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        edge = min(w, h)
        if min_edge is None or edge < min_edge:
            min_edge = edge
    return min_edge if min_edge is not None else default_radius

# 主处理函数
def process_green_areas(image_pil, green_mask, orange_mask, block_colors, large_threshold=25):
    img_draw = ImageDraw.Draw(image_pil)
    green_mask_cv = (green_mask * 255).astype(np.uint8)

    # === Step 1: 生成阻断色 mask ===
    block_mask = np.zeros_like(green_mask)
    for rgb in block_colors:
        m = extract_color_mask(image_pil, rgb, tolerance=30)
        block_mask = np.maximum(block_mask, m)

    # 去掉阻断色区域的绿色
    green_mask_cv[block_mask > 0] = 0

    # === Step 2: 获取橙色宽度作为小圆半径参考 ===
    orange_radius = get_short_edge_width_of_orange(image_pil, orange_mask)
    circle_radius = max(1.5, int(orange_radius * 0.8))

    # === Step 3: 连通域分析 ===
    contours, _ = cv2.findContours(green_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > large_threshold or h > large_threshold:
            # --- 大块绿色：保留形状并圆弧化 ---
            mask_shape = np.zeros_like(green_mask_cv)
            cv2.drawContours(mask_shape, [cnt], -1, 255, thickness=-1)

            # 圆弧化处理
            mask_shape = cv2.dilate(mask_shape, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            mask_shape = cv2.erode(mask_shape, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

            # 填充到原图（无空洞）
            ys, xs = np.where(mask_shape > 0)
            for px, py in zip(xs, ys):
                if block_mask[py, px] == 0:
                    img_draw.point((px, py), fill=(0, 152, 67))

        else:
            # --- 小块绿色：画圆 ---
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if block_mask[cy, cx] == 0:
                    bbox = [cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius]
                    img_draw.ellipse(bbox, fill=(0, 152, 67))

    return image_pil


import numpy as np
import cv2
from PIL import Image, ImageDraw

def estimate_orange_width(orange_mask):
    """
    根据橙色掩码估计线条宽度（半径）
    简单方法：利用形态学膨胀和腐蚀计算宽度的经验估计值

    参数：
        orange_mask: numpy bool数组，橙色区域掩码

    返回：
        radius: int，估计的圆半径，范围大约在2~10像素之间
    """
    if orange_mask.sum() == 0:
        return 2  # 默认最小宽度，防止除零错误

    orange_uint8 = (orange_mask * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(orange_uint8, kernel, iterations=1)
    dilated = cv2.dilate(orange_uint8, kernel, iterations=1)

    # 线宽估计：膨胀区域像素数-腐蚀区域像素数，归一化面积，经验乘2调整为半径
    width_estimate = (np.sum(dilated > 0) - np.sum(eroded > 0)) / max(1, np.count_nonzero(orange_uint8 > 0))

    # 限制范围并放大
    radius = int(max(2, min(width_estimate * 1.5, 2.5)))
    return radius


def fill_green_areas_with_circles(image, green_mask, orange_mask, blocking_mask, large_size_threshold=0.5):
    """
    用圆形拼接的方式填充绿色区域，边缘呈圆弧形
    绿色圆半径根据橙色线条宽度估计，图像间半径可变

    参数：
        image: PIL.Image对象，RGB模式
        green_mask: numpy bool数组，绿色区域掩码
        orange_mask: numpy bool数组，橙色区域掩码，用于估计线宽
        blocking_mask: numpy bool数组，阻断区域掩码，避免覆盖
        large_size_threshold: int，大区域用网格圆形填充，小区域只绘制中心圆
    返回：
        PIL.Image对象，处理后的图像
    """
    img_draw = ImageDraw.Draw(image)
    green_mask_cv = (green_mask * 255).astype(np.uint8)
    blocking_mask_cv = (blocking_mask * 255).astype(np.uint8)

    # 动态估计圆半径
    radius = estimate_orange_width(orange_mask)

    # 找到绿色区域轮廓
    contours, _ = cv2.findContours(green_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 网格步长，确保圆有重叠形成圆弧边缘
    step = max(1, int(radius * 0.3))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > large_size_threshold or h > large_size_threshold:
            xs = [p[0][0] for p in cnt]
            ys = [p[0][1] for p in cnt]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            for cx in range(xmin, xmax + 1, step):
                for cy in range(ymin, ymax + 1, step):
                    if cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0:
                        # 边界检查，防止索引越界
                        if 0 <= cy < blocking_mask_cv.shape[0] and 0 <= cx < blocking_mask_cv.shape[1]:
                            # 阻断区不绘制
                            if blocking_mask_cv[cy, cx] == 0:
                                bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
                                img_draw.ellipse(bbox, fill=(0, 152, 67))
        else:
            # 小区域直接绘制中心圆
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if 0 <= cy < blocking_mask_cv.shape[0] and 0 <= cx < blocking_mask_cv.shape[1]:
                    if blocking_mask_cv[cy, cx] == 0:
                        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
                        img_draw.ellipse(bbox, fill=(0, 152, 67))

    return image


