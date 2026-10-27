


"""运行前先说明是哪个图像来源
如python generate_.py --source results\CSDL_HX_pix2pix_R1\images
"""
# ===== 1. 标准库 =====
import os
import argparse
from pathlib import Path
from datetime import datetime

# ===== 2. 第三方库 =====
import cv2
import numpy as np
from PIL import Image

# ===== 3. 项目配置 =====
from config import (
    CYAN_COLOR,
    YELLOW_COLOR,
    MAGENTA_COLOR,
    GREEN_COLOR,
    ORANGE_COLOR,
    TOLERANCE
)

# ===== 4. 颜色提取功能 =====
from util.color_extraction import (
    extract_color_regions,
    extract_orange_region
)

# ===== 5. 掩码处理功能 =====
from util.mask_processing import (
    keep_original_if_orange,
    refine_green_mask,
    close_and_fill_mask,
    mask_to_bounding_boxes,
    normalize_line_width,
    orange_curve_from_image,
    orange_curve_connect_nearest,
    connect_road_skeleton,
    repair_and_paste_orange
)


# ===== 6. 绘制功能 =====
from util.shape_drawing import (
    draw_shape_preserve_edges,
    draw_geometric_blocks,
    smooth_mask_and_draw,
    smooth_region_edges,
    draw_orange_smooth_curve,
    paste_color_with_smooth_edges
)

# ===== 7. 植被填充功能 =====
from util.vegetation_fill import fill_green_areas_with_circles




# ===== 8. 颜色相关定义 =====
ORANGE_COLOR = (241, 145, 73)   # 橙色
CYAN_COLOR = (0, 255, 255)      # 青色
YELLOW_COLOR = (255, 255, 0)    # 黄色
TARGET_COLOR = (255, 0, 255)    # 品红色
GREEN_COLOR = (0, 152, 67)      # 绿色



OUTPUT_BASE_DIR = Path("generate_images")

# 该字典定义了每个文件夹realA中用作“阻断掩码”的特殊颜色
REAL_A_COLOR_RULES = {
    "LD_DL_pix2pix": (241, 145, 73),   # 橙色（realA中作为阻断）
    "LD_GZW_pix2pix": (255, 0, 255),   # 品红（realA中作为阻断）
    "LD_ST_pix2pix": (0, 255, 255),    # 青色（realA中作为阻断）
    "CSDL_HX_pix2pix": None            # 没有阻断颜色
}


def get_next_filename(output_dir, ext=".png"):
    files = sorted(output_dir.glob(f"*{ext}"))
    if not files:
        return f"{datetime.now().strftime('%Y-%m-%d')}_0001{ext}"
    last_num = 0
    for f in reversed(files):
        parts = f.stem.split("_")
        if parts[-1].isdigit():
            last_num = int(parts[-1])
            break
    return f"{datetime.now().strftime('%Y-%m-%d')}_{last_num+1:04d}{ext}"




def enforce_canonical_colors(img: Image.Image, color_map: dict):
    """
    将图像中所有接近 color_map key 的颜色，强制映射为对应的 value
    """
    img_np = np.array(img)
    h, w, _ = img_np.shape

    for src_color, target_color in color_map.items():
        mask = extract_color_regions(
            Image.fromarray(img_np),
            src_color,
            tolerance=0
        )
        img_np[mask] = target_color

    return Image.fromarray(img_np)






#只用城市道路生成图像处理的时候_R1_建筑、植物、道路只做平滑处理，不做其他处理
def process_CSDL_HX_pix2pix_R1_images(source_dir: Path):
    print(f"处理 CSDL_HX_pix2pix_R1，路径是 {source_dir}")
    today_str = datetime.now().strftime("%Y-%m-%d")
    folder_name = source_dir.parent.name
    output_dir = OUTPUT_BASE_DIR / folder_name / today_str
    output_dir.mkdir(parents=True, exist_ok=True)

    CYAN_COLOR = (0, 255, 255)      # 青色（水体）
    GREEN_COLOR = (0, 152, 67)      # 绿色 植物
    YELLOW_COLOR = (255, 255, 0)    # 黄色 铺装
    ORANGE_COLOR = (241, 145, 73)   # 橙色 道路
    MAGENTA_COLOR = (255, 0, 255)   # 品红（构筑物）
    Lvdi_COLOR = (0, 255, 0)   # 绿地 背景

    fakeB_files = [f for f in os.listdir(source_dir) if f.endswith(("fake_B.png", "fake_B.jpg"))]



    for fakeB_file in fakeB_files:
        fakeB_path = source_dir / fakeB_file
        realA_file = fakeB_file.replace("fake_B", "real_A")
        realA_path = source_dir / realA_file

        if not realA_path.exists():
            print(f"未找到对应 real_A 文件: {realA_file}")
            continue

        fakeB_img = Image.open(fakeB_path).convert("RGB")
        realA_img = Image.open(realA_path).convert("RGB")

        result_img = realA_img.copy()
        INVALID = 255
        h, w = fakeB_img.size[1], fakeB_img.size[0]
        label_map = np.full((h, w), INVALID, dtype=np.uint8)
        # 只把 RealA 的绿地作为“有效计算域”
        mask_valid = extract_color_regions(realA_img, Lvdi_COLOR, tolerance=TOLERANCE)
        label_map[mask_valid] = 0   # 0 = baseline green open space


        # ===== R1 Step 0：从 RealA 初始化基础语义 =====
        # RealA 中的绿色 = 连续绿地背景

        mask_green_realA = extract_color_regions(
            realA_img,
            Lvdi_COLOR,
            tolerance=TOLERANCE
        )

        label_map[mask_green_realA] = 0  # vegetation background




        # 提取黑色背景掩码
        black_mask_realA = extract_color_regions(realA_img, (0, 0, 0), tolerance=30)
        # 提取紫色背景掩码
        purple_mask_realA = extract_color_regions(realA_img, (137, 86, 161), tolerance=30)

        # 将黑色和紫色合并成统一阻断掩码
        blocking_mask_for_all = np.logical_or(black_mask_realA, purple_mask_realA)

        # 从 realA 提取橙色掩码，避免覆盖
        mask_orange = extract_color_regions(realA_img, ORANGE_COLOR, tolerance=TOLERANCE)

        # 橙色（DL）——直接复制橙色区域
        # 提取橙色掩码
        mask_orange = extract_color_regions(fakeB_img, ORANGE_COLOR, tolerance=30)

        # 避免覆盖黑色和紫色底色
        mask_orange_safe = np.logical_and(mask_orange, ~blocking_mask_for_all)

        
        # 黄色（铺装）平滑粘贴
        mask_yellow  = extract_color_regions(fakeB_img, YELLOW_COLOR,  tolerance=TOLERANCE)  # pavement

        # 2. 提取并绘制青色（水体），避开橙色阻断区，边缘圆滑
        mask_cyan    = extract_color_regions(fakeB_img, CYAN_COLOR,    tolerance=TOLERANCE)  # water


        # --- R1：绿色植物，仅保留位置，颜色强制归一 ---
        mask_green = extract_color_regions(fakeB_img, GREEN_COLOR, tolerance=TOLERANCE)

        result_np = np.array(result_img)
        result_np[mask_green] = GREEN_COLOR
        result_img = Image.fromarray(result_np)


        # --- R1：道路，仅保留位置，颜色强制归一 ---
        mask_orange = extract_color_regions(fakeB_img, ORANGE_COLOR, tolerance=TOLERANCE)

        result_np = np.array(result_img)
        result_np[mask_orange] = ORANGE_COLOR
        result_img = Image.fromarray(result_np)




        # --- 消融实验：建筑仅做原位复制（无平滑） ---

        mask_magenta = extract_color_regions(
            fakeB_img,
            MAGENTA_COLOR,
            tolerance=TOLERANCE
        ).astype(np.uint8)

        # 🔥 膨胀 1–2 像素，吃掉阴影圈
        kernel = np.ones((2,2), np.uint8)
        mask_magenta_fat = cv2.dilate(mask_magenta, kernel, iterations=1)

        result_np = np.array(result_img)
        fakeB_np = np.array(fakeB_img)

        result_np[mask_magenta_fat == 1] = MAGENTA_COLOR
        result_img = Image.fromarray(result_np)


        # 最后再检查结果图，保证没有颜色绘制到黑色或紫色背景上
        result_np = np.array(result_img)
        result_np[black_mask_realA] = (0, 0, 0)
        result_np[purple_mask_realA] = (137, 86, 161)
        result_img = Image.fromarray(result_np)
        
        # 1. 红线
        red_mask = extract_color_regions(realA_img, (255, 0, 0), tolerance=30)
        result_np = np.array(result_img)
        result_np[red_mask] = (255, 0, 0)  # 覆盖到其他所有颜色上面
        result_img = Image.fromarray(result_np)

        # —— 颜色强制归一（R1 最终保证无杂色）——
        CANONICAL_COLORS = {
            ORANGE_COLOR: ORANGE_COLOR,
            GREEN_COLOR: GREEN_COLOR,
            CYAN_COLOR: CYAN_COLOR,
            YELLOW_COLOR: YELLOW_COLOR,
            MAGENTA_COLOR: MAGENTA_COLOR,
        }


        # 顺序很重要：后写的覆盖前写的
        label_map[mask_cyan    & mask_valid] = 3        # 水
        label_map[mask_yellow  & mask_valid] = 4        # 铺装
        label_map[mask_magenta_fat & mask_valid] = 5    # 建筑
        label_map[mask_green   & mask_valid] = 2        # 植被
        label_map[mask_orange  & mask_valid] = 1        # 道路


        label_output_dir = output_dir / "label_maps"
        label_output_dir.mkdir(exist_ok=True)
        
        filename = get_next_filename(output_dir, ext=".png")
        stem = filename.replace(".png", "")

        np.save(label_output_dir / f"{stem}_label.npy", label_map)
        result_img.save(output_dir / filename)

        print(f"已处理并保存: {output_dir / filename}")







#只用城市道路生成图像处理的时候_R2
def process_CSDL_HX_pix2pix_R2_images(source_dir: Path):
    print(f"处理 CSDL_HX_pix2pix_R2，路径是 {source_dir}")
    today_str = datetime.now().strftime("%Y-%m-%d")
    folder_name = source_dir.parent.name
    output_dir = OUTPUT_BASE_DIR / folder_name / today_str
    output_dir.mkdir(parents=True, exist_ok=True)

    CYAN_COLOR = (0, 255, 255)      # 青色（水体）
    GREEN_COLOR = (0, 152, 67)      # 绿色
    YELLOW_COLOR = (255, 255, 0)    # 黄色
    ORANGE_COLOR = (241, 145, 73)   # 橙色
    MAGENTA_COLOR = (255, 0, 255)   # 品红（构筑物）

    fakeB_files = [f for f in os.listdir(source_dir) if f.endswith(("fake_B.png", "fake_B.jpg"))]

    for fakeB_file in fakeB_files:
        fakeB_path = source_dir / fakeB_file
        realA_file = fakeB_file.replace("fake_B", "real_A")
        realA_path = source_dir / realA_file

        if not realA_path.exists():
            print(f"未找到对应 real_A 文件: {realA_file}")
            continue

        fakeB_img = Image.open(fakeB_path).convert("RGB")
        realA_img = Image.open(realA_path).convert("RGB")
        result_img = realA_img.copy()

        # 提取黑色背景掩码
        black_mask_realA = extract_color_regions(realA_img, (0, 0, 0), tolerance=30)
        # 提取紫色背景掩码
        purple_mask_realA = extract_color_regions(realA_img, (137, 86, 161), tolerance=30)

        # 将黑色和紫色合并成统一阻断掩码
        blocking_mask_for_all = np.logical_or(black_mask_realA, purple_mask_realA)

        # 从 realA 提取橙色掩码，避免覆盖
        mask_orange = extract_color_regions(realA_img, ORANGE_COLOR, tolerance=TOLERANCE)

        # 橙色（DL）——直接复制橙色区域
        # 提取橙色掩码
        mask_orange = extract_color_regions(fakeB_img, ORANGE_COLOR, tolerance=60)

        # 避免覆盖黑色和紫色底色
        mask_orange_safe = np.logical_and(mask_orange, ~blocking_mask_for_all)

        
        # 黄色（铺装）平滑粘贴
        result_img = paste_color_with_smooth_edges(result_img, fakeB_img, color_rgb=YELLOW_COLOR, tolerance=TOLERANCE)

        # 2. 提取并绘制青色（水体），避开橙色阻断区，边缘圆滑
        result_img = paste_color_with_smooth_edges(result_img, fakeB_img, color_rgb=CYAN_COLOR, tolerance=TOLERANCE)


        # --- 处理绿色植物，细化掩码，用绿色圆覆盖 ---
        mask_green = extract_color_regions(fakeB_img, GREEN_COLOR, tolerance=TOLERANCE)
        # 提取fakeB中的阻断掩码（橙色、品红、青色）
        mask_cyan = extract_color_regions(fakeB_img, CYAN_COLOR, tolerance=TOLERANCE)
        mask_magenta_fakeB = extract_color_regions(fakeB_img, MAGENTA_COLOR, tolerance=TOLERANCE)

        blocking_mask = ((mask_orange + mask_magenta_fakeB + mask_cyan) > 0).astype(np.uint8)
        refined_green_mask = refine_green_mask(mask_green, blocking_mask, max_side=20)
        result_img = fill_green_areas_with_circles(
            result_img,
            refined_green_mask,           # green_mask
            mask_orange.astype(np.uint8), # orange_mask
            blocking_mask.astype(np.uint8),# blocking_mask
            large_size_threshold=40
        )

         # 橙色（DL）——直接复制橙色区域
        # 提取橙色掩码
        mask_orange = extract_color_regions(fakeB_img, ORANGE_COLOR, tolerance=40)

        # 避免覆盖黑色和紫色底色
        mask_orange_safe = np.logical_and(mask_orange, ~blocking_mask_for_all)

        # 将橙色从 fakeB_img 复制到结果图上
        result_np = np.array(result_img)
        fakeB_np = np.array(fakeB_img)
        result_np[mask_orange_safe] = fakeB_np[mask_orange_safe]
        result_img = Image.fromarray(result_np)

        # 4. 橙色（DL）——修复橙色道路并粘贴
        # 阻断掩码列表，保证不覆盖青色/绿色/紫色/黑色
        # 提取青色掩码
        cyan_mask = extract_color_regions(fakeB_img, CYAN_COLOR, tolerance=TOLERANCE)
        blocking_masks = [black_mask_realA, purple_mask_realA, mask_green, cyan_mask]

        result_img = repair_and_paste_orange(
            result_img,
            fakeB_img,
            orange_rgb=(241,145,73),
            tolerance=30,
            max_width=2,
            blocking_masks=blocking_masks
        )
        


        # --- 处理品红构筑物，绘制矩形块 ---
        mask_magenta = extract_color_regions(fakeB_img, MAGENTA_COLOR, tolerance=TOLERANCE)
        # 避免覆盖橙色
        mask_magenta = np.logical_and(mask_magenta, ~mask_orange)
        magenta_mask_cv = (mask_magenta * 255).astype(np.uint8)
        boxes_magenta = mask_to_bounding_boxes(magenta_mask_cv)
        result_img = draw_geometric_blocks(result_img, boxes_magenta, MAGENTA_COLOR)


        # 最后再检查结果图，保证没有颜色绘制到黑色或紫色背景上
        result_np = np.array(result_img)
        result_np[black_mask_realA] = (0, 0, 0)
        result_np[purple_mask_realA] = (137, 86, 161)
        result_img = Image.fromarray(result_np)
        
        # 1. 红线
        red_mask = extract_color_regions(realA_img, (255, 0, 0), tolerance=30)
        result_np = np.array(result_img)
        result_np[red_mask] = (255, 0, 0)  # 覆盖到其他所有颜色上面
        result_img = Image.fromarray(result_np)


        # —— 颜色强制归一（R2 最终保证无杂色）——
        CANONICAL_COLORS = {
            ORANGE_COLOR: ORANGE_COLOR,
            GREEN_COLOR: GREEN_COLOR,
            CYAN_COLOR: CYAN_COLOR,
            YELLOW_COLOR: YELLOW_COLOR,
            MAGENTA_COLOR: MAGENTA_COLOR,
        }

        result_img = enforce_canonical_colors(result_img, CANONICAL_COLORS)


        filename = get_next_filename(output_dir, ext=".png")
        result_img.save(output_dir / filename)
        print(f"已处理并保存: {output_dir / filename}")



import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="根据输入路径处理图像")
    parser.add_argument(
        "--source",
        required=True,
        help="输入图像文件夹路径，例如 results/LD_GZW_pix2pix_R1/images"
    )

    args = parser.parse_args()
    source_dir = Path(args.source)

    if not source_dir.exists():
        raise FileNotFoundError(f"输入路径不存在: {source_dir}")

    # 读取上一级文件夹名（如 CSDL_HX_pix2pix_R1）
    folder_name = source_dir.parent.name

    if folder_name == "CSDL_HX_pix2pix_R1":
        process_CSDL_HX_pix2pix_R1_images(source_dir)
    elif folder_name == "CSDL_HX_pix2pix_R2":
        process_CSDL_HX_pix2pix_R2_images(source_dir)
    else:
        print(f"不支持的文件夹类型: {folder_name}")


if __name__ == "__main__":
    main()
