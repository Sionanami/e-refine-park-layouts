# 绘制形状相关
import numpy as np
import cv2
from config import ORANGE_COLOR, TOLERANCE_ORANGE
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree



def paste_color_with_smooth_edges(real_img: Image.Image, source_img: Image.Image, color_rgb: tuple, tolerance=30, kernel_size=5, smoothing_points=100):
    """
    从 source_img 提取指定颜色区域，并粘贴到 real_img 上，边缘为平滑曲线。

    参数：
        real_img: PIL.Image，要粘贴到的图像
        source_img: PIL.Image，要提取颜色的图像
        color_rgb: tuple, 目标颜色 RGB
        tolerance: 色差容差
        kernel_size: 形态学闭运算核大小，用于平滑掩码
        smoothing_points: 拟合曲线的采样点数量

    返回：
        PIL.Image，粘贴后的图像
    """
    # 转为 numpy
    src_np = np.array(source_img)
    real_np = np.array(real_img)

    # 计算色差
    diff = np.linalg.norm(src_np - np.array(color_rgb), axis=2)
    mask = diff <= tolerance

    # 形态学闭运算，去除小孔和噪点
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # 找轮廓
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        if len(cnt) < 3:
            continue
        # 提取轮廓点
        pts = np.array([p[0] for p in cnt])
        # 闭合轮廓
        pts = np.vstack([pts, pts[0]])

        # B样条曲线拟合
        try:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=1.0, per=True)
            u_new = np.linspace(0, 1, smoothing_points)
            x_new, y_new = splev(u_new, tck)
            smooth_pts = np.vstack([x_new, y_new]).T.astype(np.int32)
        except:
            # 拟合失败就使用原始轮廓
            smooth_pts = pts.astype(np.int32)

        # 创建掩码并填充
        mask_draw = np.zeros(real_np.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_draw, [smooth_pts], 255)

        # 粘贴颜色
        real_np[mask_draw > 0] = color_rgb

    return Image.fromarray(real_np)



def smooth_region_edges(image: Image.Image, mask: np.ndarray, color: tuple, kernel_size=5, epsilon_factor=0.01):
    """
    黄色/青色区域平滑填充
    image: PIL.Image
    mask: np.bool掩码
    color: RGB颜色
    kernel_size: 形态学闭运算核大小
    epsilon_factor: approxPolyDP精度
    """
    mask_cv = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask_cv, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = ImageDraw.Draw(image)
    for cnt in contours:
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) >= 3:
            draw.polygon(pts, fill=color)
    return image


def draw_orange_smooth_curve(image: Image.Image, mask: np.ndarray, color: tuple, line_width=1, max_dist=3, smoothing=1):
    """
    橙色区域骨架化+曲线拟合绘制平滑线条
    image: PIL.Image
    mask: np.bool掩码
    color: RGB
    line_width: 线宽
    max_dist: 就近连接点阈值
    smoothing: B样条平滑参数
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    skeleton = skeletonize(mask_uint8 > 0).astype(np.uint8)
    ys, xs = np.where(skeleton > 0)
    points = np.vstack([xs, ys]).T
    if len(points) < 2:
        return image

    tree = cKDTree(points)
    visited = set()
    paths = []

    for i in range(len(points)):
        if i in visited:
            continue
        path = [points[i]]
        visited.add(i)
        current_idx = i
        while True:
            dists, idxs = tree.query(points[current_idx], k=20, distance_upper_bound=max_dist)
            candidates = [idx for dist, idx in zip(dists, idxs) if dist != np.inf and idx not in visited and idx != current_idx]
            if candidates:
                next_idx = candidates[0]
                path.append(points[next_idx])
                visited.add(next_idx)
                current_idx = next_idx
            else:
                break
        path = np.array(path)
        if len(path) > 2:
            path = path[path[:,0].argsort()]  # 按x排序保证平滑
        paths.append(path)

    # 转PIL绘图
    draw = ImageDraw.Draw(image)
    for path_pts in paths:
        if len(path_pts) < 2:
            continue
        # B样条拟合
        if len(path_pts) > 3:
            tck, u = splprep(path_pts.T, s=smoothing, k=min(3, len(path_pts)-1))
            u_fine = np.linspace(0, 1, max(500, len(path_pts)*5))
            x_fine, y_fine = splev(u_fine, tck)
            smooth_pts = list(zip(x_fine.astype(int), y_fine.astype(int)))
        else:
            smooth_pts = [tuple(pt) for pt in path_pts]
        draw.line(smooth_pts, fill=color, width=line_width)
    return image




def smooth_mask_and_draw(image, mask, color, kernel_size=7, epsilon_factor=0.02):
    mask_cv = (mask * 255).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_cv, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = ImageDraw.Draw(image)
    for cnt in contours:
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        if len(pts) >= 3:
            draw.polygon(pts, fill=color)
    return image

def draw_geometric_blocks(image, boxes, color, max_area=10000):
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in boxes:
        area = w * h
        if area > max_area:
            continue
        draw.rectangle([x, y, x + w, y + h], fill=tuple(color))
    return image

def draw_shape_preserve_edges(image, mask, color, min_area=10):
    mask_cv = (mask * 255).astype(np.uint8)
    # 形态学闭运算，核大小可以调整
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_cv = cv2.morphologyEx(mask_cv, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = ImageDraw.Draw(image)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # 这里可以用 approxPolyDP 平滑轮廓点，调节精度epsilon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) >= 3:
            draw.polygon(pts, fill=color)
    return image


import cv2
import numpy as np
from PIL import ImageDraw


def keep_original_if_orange(realA_img, mask):
    realA_np = np.array(realA_img)
    dist = np.linalg.norm(realA_np - np.array(ORANGE_COLOR), axis=2)
    orange_mask = dist < TOLERANCE_ORANGE
    mask[orange_mask] = 0
    return mask

def refine_green_mask(green_mask, blocking_mask, max_side=50):
    refined_mask = green_mask.copy()
    refined_mask[blocking_mask > 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
    new_mask = np.zeros_like(refined_mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        component_mask = (labels == i).astype(np.uint8)
        if w > max_side or h > max_side:
            h_splits = int(np.ceil(h / max_side))
            w_splits = int(np.ceil(w / max_side))
            for hs in range(h_splits):
                for ws in range(w_splits):
                    sub_x = x + ws * max_side
                    sub_y = y + hs * max_side
                    sub_w = min(max_side, x + w - sub_x)
                    sub_h = min(max_side, y + h - sub_y)
                    sub_mask = component_mask[sub_y:sub_y + sub_h, sub_x:sub_x + sub_w]
                    if np.any(sub_mask):
                        new_mask[sub_y:sub_y + sub_h, sub_x:sub_x + sub_w] = np.maximum(
                            new_mask[sub_y:sub_y + sub_h, sub_x:sub_x + sub_w], sub_mask)
        else:
            new_mask[y:y+h, x:x+w] = np.maximum(new_mask[y:y+h, x:x+w], component_mask[y:y+h, x:x+w])
    return new_mask
