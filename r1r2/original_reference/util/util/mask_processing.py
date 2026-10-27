# mask优化、阻断、平滑处理
import numpy as np
import cv2
from config import ORANGE_COLOR, TOLERANCE_ORANGE
from PIL import Image
from collections import deque
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree





def repair_and_paste_orange(realA_img, fakeB_img, orange_rgb=(241,145,73),
                             tolerance=40, max_width=2, blocking_masks=None):
    """
    修复橙色断点并粘贴到 realA 图像上（两次复制）
    第二层骨架连接可参考特定颜色，但不复制或膨大
    """
    import numpy as np
    import cv2
    from skimage.morphology import skeletonize
    from scipy.spatial import cKDTree
    from PIL import Image

    realA_np = np.array(realA_img)
    fakeB_np = np.array(fakeB_img)

    # ------------------ 第一次复制橙色 ------------------
    lower = np.array([max(0, c - tolerance) for c in orange_rgb], dtype=np.uint8)
    upper = np.array([min(255, c + tolerance) for c in orange_rgb], dtype=np.uint8)
    mask_orange = cv2.inRange(fakeB_np, lower, upper) > 0  # bool掩码

    result_np = realA_np.copy()

    # 阻断区域
    if blocking_masks:
        combined_block = np.zeros_like(mask_orange, dtype=bool)
        for m in blocking_masks:
            combined_block = np.logical_or(combined_block, m)
        mask_orange_safe = np.logical_and(mask_orange, ~combined_block)
    else:
        mask_orange_safe = mask_orange

    # 第一次粘贴基础层
    result_np[mask_orange_safe] = orange_rgb

    # ------------------ 第二层骨架化 + 断点补全 ------------------
    # 单独提取橙色掩码
    mask_layer = mask_orange.astype(np.uint8)
    skeleton = skeletonize(mask_layer).astype(np.uint8)

    # 定义参考颜色，只作靠近参考使用
    reference_colors = [(255,0,255), (255,255,0), (0,152,67), (241,145,73), (137,86,161)]
    reference_mask = np.zeros(fakeB_np.shape[:2], dtype=bool)
    for color in reference_colors:
        lower_ref = np.array([max(0,c-10) for c in color], dtype=np.uint8)
        upper_ref = np.array([min(255,c+10) for c in color], dtype=np.uint8)
        mask_ref = cv2.inRange(fakeB_np, lower_ref, upper_ref) > 0
        reference_mask = np.logical_or(reference_mask, mask_ref)

    # 端点检测
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = np.argwhere(filtered >= 4)

    # 连通域标记孤立段
    num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))

    # 所有骨架点
    points = np.argwhere(skeleton > 0)

    if len(points) > 1 and len(endpoints) > 1:
        tree = cKDTree(points)
        for y, x in endpoints:
            # 当前端点所在段 ID
            current_label = labels[y, x]
            connected_count = 0  # 计数器：当前端点已连接次数

            # 查询较多近邻，但不全用
            max_neighbors = min(20, len(points))  # 捕捉范围
            dist, idx = tree.query([y, x], k=max_neighbors)

            for i in range(1, len(idx)):
                nearest_y, nearest_x = points[idx[i]]
                nearest_label = labels[nearest_y, nearest_x]

                # 只连接不同孤立段
                if nearest_label == current_label:
                    continue

                # 画线连接
                cv2.line(skeleton, (x, y), (nearest_x, nearest_y), 1, thickness=1)
                connected_count += 1

                # 最多连接 2 次
                if connected_count >= 3:
                    break


    # ------------------ 宽度控制 ------------------
    skeleton_uint8 = (skeleton*255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_width, max_width))
    skeleton_dilated = cv2.dilate(skeleton_uint8, kernel)
    skeleton_final = cv2.erode(skeleton_dilated, kernel)

    # ------------------ 避免覆盖阻断区 ------------------
    if blocking_masks:
        skeleton_final = np.logical_and(skeleton_final>0, ~combined_block)

    # ------------------ 粘贴第二层橙色 ------------------
    result_np[skeleton_final>0] = orange_rgb

    return Image.fromarray(result_np)




# def repair_and_paste_orange(realA_img, fakeB_img, orange_rgb=(241,145,73), tolerance=40...
#     """
#     修复橙色断点并粘贴到 realA 图像上（两层复制）
    
#     参数：
#         realA_img: PIL.Image, RGB模式
#         fakeB_img: PIL.Image, RGB模式
#         orange_rgb: 橙色 RGB
#         tolerance: 提取橙色时的容差
#         max_width: 最大宽度，超过就切掉
#         blocking_masks: list of np.ndarray, 布尔掩码列表，避免覆盖
#     返回：
#         PIL.Image, 处理后的图像
#     """
#     # 转为 numpy
#     realA_np = np.array(realA_img)
#     fakeB_np = np.array(fakeB_img)

#     # ------------------ 第一次复制橙色（基础层） ------------------
#     lower = np.array([max(0, c - tolerance) for c in orange_rgb], dtype=np.uint8)
#     upper = np.array([min(255, c + tolerance) for c in orange_rgb], dtype=np.uint8)
#     mask_orange = cv2.inRange(fakeB_np, lower, upper)  # uint8 0/255
#     mask_orange_bool = mask_orange > 0

#     result_np = realA_np.copy()
    
#     # 避免覆盖阻断区域
#     if blocking_masks:
#         combined_block = np.zeros_like(mask_orange_bool)
#         for m in blocking_masks:
#             combined_block = np.logical_or(combined_block, m)
#         mask_orange_safe = np.logical_and(mask_orange_bool, ~combined_block)
#     else:
#         mask_orange_safe = mask_orange_bool

#     # 第一次粘贴橙色基础层
#     result_np[mask_orange_safe] = orange_rgb

#     # ------------------ 第二次复制橙色（骨架+断点补全） ------------------
#     mask_layer = mask_orange_safe.astype(np.uint8)  # 0/1
#     skeleton = skeletonize(mask_layer > 0).astype(np.uint8)

#     # 检测骨架端点
#     kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
#     filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
#     endpoints = np.argwhere(filtered == 11)  # 骨架端点

#     points = np.argwhere(skeleton > 0)
#     if len(points) > 1 and len(endpoints) > 1:
#         tree = cKDTree(points)
#         for y, x in endpoints:
#             dist, idx = tree.query([y, x], k=2)  # 最近的点是自己
#             nearest_y, nearest_x = points[idx[1]]
#             # 用 cv2.line 画线连接断点
#             line_img = np.zeros_like(skeleton, dtype=np.uint8)
#             cv2.line(line_img, (x, y), (nearest_x, nearest_y), 3, 3)
#             # 将线段加入 skeleton
#             py_px = np.argwhere(line_img > 0)
#             for py, px in py_px:
#                 skeleton[py, px] = 1

#     # ------------------ 宽度控制 ------------------
#     skeleton_uint8 = (skeleton * 255).astype(np.uint8)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_width, max_width))
#     skeleton_dilated = cv2.dilate(skeleton_uint8, kernel)
#     skeleton_final = cv2.erode(skeleton_dilated, kernel)

#     # 避免覆盖阻断区域
#     if blocking_masks:
#         skeleton_final = np.logical_and(skeleton_final > 0, ~combined_block).astype(np.uint8)

#     # 粘贴第二层橙色到 result_np
#     mask_final = skeleton_final > 0
#     result_np[mask_final] = orange_rgb

#     return Image.fromarray(result_np)




# ===============================
# 辅助函数
# ===============================
def find_endpoints(component_mask):
    """
    返回骨架端点 [(x1,y1), ...]
    """
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    conv = cv2.filter2D(component_mask.astype(np.uint8), -1, kernel)
    endpoints = np.argwhere(conv==255+10)  # 中心10 + 周围1个像素=255
    return [(x,y) for y,x in endpoints]

def shortest_path_between_points(mask, pt1, pt2):
    """
    在二值掩码中找到 pt1 到 pt2 的最短路径（BFS）
    返回路径列表 [(x,y), ...]
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    prev = np.full((h,w,2), -1, dtype=int)

    queue = deque([pt1])
    visited[pt1[1], pt1[0]] = True
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]  # 8邻域
    found = False

    while queue and not found:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0<=nx<w and 0<=ny<h and not visited[ny,nx]:
                visited[ny,nx] = True
                prev[ny,nx] = [x,y]
                queue.append((nx,ny))
                if (nx,ny)==pt2:
                    found = True
                    break

    # 回溯路径
    path = []
    if found:
        x, y = pt2
        while (x,y) != pt1:
            path.append((x,y))
            x, y = prev[y,x]
        path.append(pt1)
        path.reverse()
    return path




#道路相关
def connect_road_skeleton(mask_orange, max_connect_dist=50):
    """
    mask_orange: 二值化道路掩码（bool）
    max_connect_dist: 允许连接的最大像素距离
    返回: 修复后的骨架掩码
    """
    # 1. 骨架化
    skeleton = skeletonize(mask_orange).astype(np.uint8)

    # 2. 提取骨架端点（只有一个邻居的像素）
    from scipy.ndimage import convolve

    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    conv = convolve(skeleton, np.ones((3,3)), mode='constant')
    endpoints = np.where((skeleton==1) & (conv<=2))  # 简单近似端点

    endpoints_list = list(zip(endpoints[0], endpoints[1]))

    # 3. 对每个端点寻找最近端点并连接
    from scipy.spatial import distance
    connected = skeleton.copy()

    for i, pt1 in enumerate(endpoints_list):
        min_dist = max_connect_dist
        nearest_pt = None
        for j, pt2 in enumerate(endpoints_list):
            if i >= j:
                continue
            d = distance.euclidean(pt1, pt2)
            if d < min_dist:
                min_dist = d
                nearest_pt = pt2
        if nearest_pt is not None:
            # 用线段连接两端点
            cv2.line(connected, (pt1[1], pt1[0]), (nearest_pt[1], nearest_pt[0]), 1, 1)

    return connected.astype(bool)





def keep_original_if_orange(realA_img, mask):
    realA_np = np.array(realA_img)
    dist = np.linalg.norm(realA_np - np.array(ORANGE_COLOR), axis=2)
    orange_mask = dist < TOLERANCE_ORANGE
    mask[orange_mask] = 0
    return mask

import cv2
import numpy as np

def refine_green_mask(green_mask, blocking_mask, max_side=50):
    # 复制输入掩码并去除阻断区域
    refined_mask = green_mask.copy()
    refined_mask[blocking_mask > 0] = 0

    # 转为 uint8 类型，确保cv2函数支持
    refined_mask = refined_mask.astype(np.uint8)

    # 3x3椭圆核进行开运算（去噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # 连通组件分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)

    # 新掩码，初始化为全0，类型为uint8
    new_mask = np.zeros_like(refined_mask, dtype=np.uint8)

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
            new_mask[y:y + h, x:x + w] = np.maximum(new_mask[y:y + h, x:x + w], component_mask[y:y + h, x:x + w])

    return new_mask



import cv2
import numpy as np

def close_and_fill_mask(mask_bool, kernel_size=15):
    """
    对bool掩码做形态学闭运算，连接断裂线条
    """
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    # 转回bool
    return closed.astype(bool)

def mask_to_bounding_boxes(mask_uint8):
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes



def normalize_line_width(mask_bool, target_width=5):
    """
    规范二值掩码中线条的宽度，使所有线条宽度大致相等。

    参数：
        mask_bool: numpy bool数组，二值掩码
        target_width: int，目标线宽，单位像素

    返回：
        numpy bool数组，宽度规范后的掩码
    """
    mask_uint8 = mask_bool.astype(np.uint8) * 255

    # 先腐蚀，去掉细小多余宽度（可调整迭代次数）
    erode_iter = max(1, target_width // 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    eroded = cv2.erode(mask_uint8, kernel, iterations=erode_iter)

    # 再膨胀，恢复到目标宽度
    dilate_iter = max(1, target_width // 2)
    normalized = cv2.dilate(eroded, kernel, iterations=dilate_iter)

    return normalized.astype(bool)

import cv2
import numpy as np
from skimage.morphology import skeletonize

import numpy as np

def extract_color_regions(image_pil, target_rgb, tolerance=30):
    """
    从PIL图像中提取与目标颜色接近的区域掩码。

    参数：
        image_pil: PIL.Image 对象，RGB模式
        target_rgb: tuple(int,int,int)，目标颜色RGB值
        tolerance: int，颜色容差，默认30

    返回：
        numpy bool数组，掩码区域为True
    """
    img_np = np.array(image_pil)
    dist = np.linalg.norm(img_np - np.array(target_rgb), axis=2)
    mask = dist <= tolerance
    return mask


def smooth_orange_line_from_mask(fakeB_img, orange_rgb, tolerance=30, line_width=5):
    # 1. 提取橙色掩码
    mask_orange = extract_color_regions(fakeB_img, orange_rgb, tolerance=tolerance)

    # 2. 形态学闭合，补全断裂
    mask_orange = close_and_fill_mask(mask_orange, kernel_size=15)

    # 3. 细化骨架
    skeleton = skeletonize(mask_orange).astype(np.uint8) * 255

    # 4. 找骨架轮廓（曲线）
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 5. 创建空白图绘制线条
    blank = np.zeros(mask_orange.shape, dtype=np.uint8)

    for cnt in contours:
        if len(cnt) > 1:
            epsilon = 0.01 * cv2.arcLength(cnt, False)  # 调整平滑程度
            approx = cv2.approxPolyDP(cnt, epsilon, False)
            pts = approx.reshape(-1, 2)
            cv2.polylines(blank, [pts], isClosed=False, color=255, thickness=line_width)

    # 返回bool掩码
    return blank.astype(bool)


import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev

def fit_smooth_curve(points, smoothing=0):
    """
    用B样条拟合平滑曲线
    points: Nx2数组
    smoothing: 平滑参数，0是插值，越大越平滑
    返回：平滑曲线上的点数组
    """
    tck, u = splprep(points.T, s=smoothing)
    u_fine = np.linspace(0, 1, max(200, len(points)*3))
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack([x_fine, y_fine]).T.astype(np.int32)

def orange_curve_from_image(img_pil, orange_rgb, tolerance=5, line_width=0.5, smoothing=20):
    # 1. 提取橙色掩码
    img_np = np.array(img_pil)
    dist = np.linalg.norm(img_np - np.array(orange_rgb), axis=2)
    mask = dist <= tolerance

    # 2. 闭运算补全
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # 3. 骨架化
    skeleton = skeletonize(mask_closed > 0).astype(np.uint8)

    # 4. 获取骨架坐标点
    ys, xs = np.where(skeleton > 0)
    points = np.vstack([xs, ys]).T
    if len(points) < 2:
        return np.zeros_like(mask_uint8)  # 无骨架点，返回空图

    # 5. 按x排序，防止曲线跳跃（简单处理，实际可用图论寻找路径）
    points = points[points[:,0].argsort()]

    # 6. 拟合平滑曲线
    curve_pts = fit_smooth_curve(points, smoothing=smoothing)

    # 7. 画线
    blank = np.zeros_like(mask_uint8)
    cv2.polylines(blank, [curve_pts], isClosed=False, color=255, thickness=line_width, lineType=cv2.LINE_AA)

    return blank.astype(bool)


import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

def fit_smooth_curve(points, smoothing=0):
    m = len(points)
    if m < 2:
        return points.astype(np.int32)
    k = 3
    if m <= k:
        k = m - 1
    tck, u = splprep(points.T, s=smoothing, k=k)
    u_fine = np.linspace(0, 1, max(500, m*5))
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack([x_fine, y_fine]).T.astype(np.int32)

def orange_curve_connect_nearest(fakeB_img, orange_rgb, tolerance=15, line_width=2, smoothing=1, max_dist=5):
    """
    识别fakeB_img中的橙色区域，骨架化后按就近距离(max_dist)连线，
    用B样条平滑曲线绘制等宽线条，返回bool掩码。

    参数：
        fakeB_img: np.ndarray 或 PIL.Image，RGB图像
        orange_rgb: tuple，目标橙色RGB
        tolerance: 色差容差，默认15（调大）
        line_width: 线宽像素，默认2
        smoothing: 曲线平滑度参数，默认1
        max_dist: 连接点最大距离阈值，默认5像素

    返回：
        np.bool数组，橙色曲线掩码
    """
    if not isinstance(fakeB_img, np.ndarray):
        img_np = np.array(fakeB_img)
    else:
        img_np = fakeB_img

    # 1. 提取橙色掩码，容差较大
    dist = np.linalg.norm(img_np - np.array(orange_rgb), axis=2)
    mask = dist <= tolerance
    mask_uint8 = mask.astype(np.uint8) * 255

    # 2. 膨胀，让小橙色块稍微变大，避免太小被忽略
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_dilated = cv2.dilate(mask_uint8, kernel_dilate, iterations=1)

    # 3. 闭运算补全断裂
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel_close)

    # 4. 轻微腐蚀细化，保证骨架单线（注意腐蚀力度不要太大）
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_refined = cv2.erode(mask_closed, kernel_erode, iterations=1)

    # 5. 骨架化
    skeleton = skeletonize(mask_refined > 0).astype(np.uint8)
    ys, xs = np.where(skeleton > 0)
    points = np.vstack([xs, ys]).T

    if len(points) < 2:
        return np.zeros_like(mask_uint8).astype(bool)

    # 6. 用KD树做就近连线，避免四面八方乱连，最大距离阈值max_dist控制连线范围
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
        # 按x轴排序，保持曲线平滑连续
        path = path[path[:, 0].argsort()]
        paths.append(path)

    # 7. 在空白图上画平滑曲线
    blank = np.zeros_like(mask_uint8)

    for path_pts in paths:
        if len(path_pts) > 2:
            smooth_pts = fit_smooth_curve(path_pts, smoothing)
        else:
            smooth_pts = path_pts.astype(np.int32)
        cv2.polylines(blank, [smooth_pts], isClosed=False, color=255, thickness=line_width, lineType=cv2.LINE_AA)

    # 8. 可选膨胀一次让线条连贯
    kernel_dilate2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    blank = cv2.dilate(blank, kernel_dilate2, iterations=1)

    return blank.astype(bool)



def close_and_fill_mask(mask_bool, kernel_size=15):
    """
    形态学闭运算，连接断裂线条
    """
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)

def extract_color_regions(image_pil, target_rgb, tolerance=30):
    """
    从PIL图像中提取与目标颜色接近的区域掩码。
    """
    img_np = np.array(image_pil)
    dist = np.linalg.norm(img_np - np.array(target_rgb), axis=2)
    mask = dist <= tolerance
    return mask

def normalize_line_width(mask_bool, target_width=3):
    """
    规范二值掩码线条宽度为目标宽度
    """
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    erode_iter = max(1, target_width // 3)
    dilate_iter = max(1, target_width // 2)
    eroded = cv2.erode(mask_uint8, kernel, iterations=erode_iter)
    normalized = cv2.dilate(eroded, kernel, iterations=dilate_iter)
    return normalized.astype(bool)
