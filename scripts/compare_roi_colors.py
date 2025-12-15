import colorsys

import matplotlib.pyplot as plt
import numpy as np


# ===============================
# 颜色生成方法
# ===============================

# 方法 1：固定步长（原始方案）
def color_fixed_step(i):
    return [(i * 37) % 256, (i * 59) % 256, (i * 83) % 256]


# 方法 2：HSV 均匀分布（依赖 ROI 数量）
def color_hsv_uniform(i, num_rois, s=0.9, v=0.9):
    h = i / max(1, num_rois)  # 0~1
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(255 * r), int(255 * g), int(255 * b)]


# 方法 3：黄金角（不依赖 ROI 数量）
GOLDEN_RATIO = 0.618033988749895


def color_golden_ratio(i, s=0.8, v=0.95):
    h = (i * GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(255 * r), int(255 * g), int(255 * b)]


if __name__ == '__main__':
    NUM_ROIS = 20
    indices = np.arange(NUM_ROIS)

    # 归一化到 [0,1] 供 matplotlib 使用
    colors_fixed = [np.array(color_fixed_step(i)) / 255.0 for i in indices]
    colors_hsv = [np.array(color_hsv_uniform(i, NUM_ROIS)) / 255.0 for i in indices]
    colors_gold = [np.array(color_golden_ratio(i)) / 255.0 for i in indices]

    # ===============================
    # 可视化
    # ===============================
    plt.figure(figsize=(12, 4))

    plt.scatter(indices, np.ones_like(indices) * 3, c=colors_fixed, s=200)
    plt.scatter(indices, np.ones_like(indices) * 2, c=colors_hsv, s=200)
    plt.scatter(indices, np.ones_like(indices) * 1, c=colors_gold, s=200)

    plt.yticks(
        [1, 2, 3],
        ["Golden Ratio", "HSV Uniform (by ROI count)", "Fixed Step"]
    )

    plt.xlabel("ROI Index")
    plt.title("Comparison of ROI Color Generation Strategies")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.ylim(0.5, 3.5)

    plt.tight_layout()
    plt.show()
