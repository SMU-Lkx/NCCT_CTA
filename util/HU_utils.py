# coding=utf-8
# util/HU_utils.py
"""
CT值相关方法。
"""
import numpy as np
import torch
from matplotlib import colors
from matplotlib import cm


def foreground_ratio(hu_slice: np.ndarray, air_thresh: int = -100):
    """
    计算前景占比（非空气像素比例）
    hu_slice: 2-D HU 值 numpy 数组（float32）
    air_thresh: HU <= 该值视为空气
    返回: float 比例 0~1
    """
    fg_mask = hu_slice > air_thresh
    return fg_mask.sum() / fg_mask.size


def hu2rgb(hu, vmin, vmax):
    """
    HU转换为RGB图像。带伪彩。
    """
    if hu.ndim == 4:
        hu = hu.squeeze(1)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('seismic')
    rgba = cmap(norm(hu.cpu().numpy()))  # (B,H,W,4)
    rgb = torch.from_numpy(rgba[..., :3]).permute(0, 3, 1, 2).float()  # (B,3,H,W)
    return rgb


def hu2uint8(img, HU_WW, HU_WL):
    """HU -> 0-255 uint8"""
    low, high = HU_WL - HU_WW//2, HU_WL + HU_WW//2
    img = np.clip(img, low, high)
    return ((img - low) / (high - low + 1e-7) * 255).astype(np.uint8)


def denorm_to_hu(x, args):
    hu = x.float() * args.norm_factor
    return hu.clamp(args.hu_min, args.hu_max)