# coding=utf-8
# Transforms.py
"""
不使用torchvision的图像预处理。
"""
import numpy as np
from PIL import Image, ImageOps
from albumentations.core.transforms_interface import ImageOnlyTransform
import torch


# -------- 预处理管道组合 --------
class Compose:
    """轻量级预处理管道组合器，完全替代 torchvision.transforms.Compose"""
    def __init__(self, transforms):
        """
        初始化预处理管道
        :param transforms: 预处理函数或可调用对象组成的列表
        """
        self.transforms = transforms

    def __call__(self, img):
        """顺序应用所有预处理函数"""
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        """返回管道的可读表示"""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


# -------- 调整窗宽窗位（albumentations） --------
class AdjustWindow_A(ImageOnlyTransform):
    def __init__(self, window_width, window_center, always_apply=True):
        """
        :param window_width: 窗宽
        :param window_center: 窗位
        """
        super().__init__(always_apply)
        self.window_width = window_width
        self.window_center = window_center

    def apply(self, image, **params):
        """
        基于NumPy数据调整窗宽窗位
        :param image: NumPy数组格式的原始像素数据
        :return: 调整后的图像(numpy格式)
        """
        min_value = self.window_center - (self.window_width / 2)
        max_value = self.window_center + (self.window_width / 2)

        # 确保图像是浮点类型以便计算
        image = image.astype(np.float32)

        # 截断超出窗宽范围的值
        adjusted_image = np.clip(image, min_value, max_value)

        # 归一化到0-255
        adjusted_image = ((adjusted_image - min_value) / self.window_width) * 255

        # 转换为uint8并保持单通道
        return adjusted_image.astype(np.uint8)

    def get_transform_init_args_names(self):
        return "window_width", "window_center"


# -------- 中心裁剪 --------
class CenterCrop:
    """轻量级中心裁剪实现，完美替代 transforms.CenterCrop，支持中文路径"""
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0):
        """
        初始化中心裁剪器
        :param size: 目标尺寸 (int 或 (h, w) 元组)
        :param padding: 可选填充 (int 或 (pad_h, pad_w) 元组)
        :param pad_if_needed: 如果图像小于目标尺寸是否自动填充
        :param fill: 填充颜色 (int 或 (R, G, B) 元组)
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        if padding is not None:
            if isinstance(padding, int):
                self.padding = (padding, padding, padding, padding)  # 左, 上, 右, 下
            elif len(padding) == 2:
                self.padding = (padding[1], padding[0], padding[1], padding[0])  # (水平, 垂直) -> (左, 上, 右, 下)
            else:
                self.padding = padding
        else:
            self.padding = None

        self.pad_if_needed = pad_if_needed
        self.fill = fill

    def __call__(self, img):
        """
        应用中心裁剪
        :param img: PIL.Image 图像
        :return: 裁剪后的图像
        """
        # 处理填充逻辑
        if self.padding is not None:
            img = self._apply_padding(img)

        # 获取图像尺寸
        width, height = img.size

        # 检查是否需要自动填充
        if self.pad_if_needed:
            if height < self.size[0] or width < self.size[1]:
                img = self._apply_padding(img, auto_pad=True)
                width, height = img.size

        # 计算裁剪区域
        crop_width, crop_height = self.size
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # 应用裁剪
        return img.crop((left, top, right, bottom))

    def _apply_padding(self, img, auto_pad=False):
        """应用填充到图像"""
        if auto_pad:
            width, height = img.size
            pad_horizontal = max(0, self.size[0] - width)
            pad_vertical = max(0, self.size[1] - height)

            # 计算对称填充
            left = pad_horizontal // 2
            top = pad_vertical // 2
            right = pad_horizontal - left
            bottom = pad_vertical - top

            padding = (left, top, right, bottom)
        else:
            padding = self.padding

        # 应用填充
        if img.mode == 'RGBA':
            # 处理透明通道的特殊情况
            return self._pad_rgba_image(img, padding)
        else:
            return ImageOps.expand(img, border=padding, fill=self.fill)

    def _pad_rgba_image(self, img, padding):
        """处理带透明通道的图像填充"""
        # 创建新图像
        new_width = img.width + padding[0] + padding[2]
        new_height = img.height + padding[1] + padding[3]
        new_img = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))

        # 粘贴原始图像
        new_img.paste(img, (padding[0], padding[1]))

        # 如果指定了填充颜色，需要处理非透明情况
        if self.fill != (0, 0, 0, 0):
            # 创建填充层
            fill_layer = Image.new('RGBA', new_img.size, self.fill)
            # 合并填充层和图像
            new_img = Image.alpha_composite(fill_layer, new_img)

        return new_img

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding}, pad_if_needed={self.pad_if_needed}, fill={self.fill})"


# -------- 最大密度投影图 --------
class MIP:
    def __init__(self, axis=2):
        """
        初始化最大密度投影类
        :param axis: 沿哪个轴进行投影，默认为0（即Z轴）
        """
        self.axis = axis

    def __call__(self, image):
        """
        对输入的3D图像进行最大密度投影
        :param image: 输入的3D图像，形状为 (depth, height, width)
        :return: 投影后的2D图像，形状为 (height, width)
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("输入图像必须是3D NumPy数组")

        # 沿指定轴进行最大密度投影
        projected_image = np.max(image, axis=self.axis)

        return projected_image


# -------- 非目标CT值置0 --------
class Seg2Zero:
    def __init__(self, hu_min, hu_max, threshold=200, target='higher' or 'lower'):
        """
        Args:
            threshold (int): HU值阈值，高于/低于此值的像素被视为目标组织(骨头约200以上)
        """
        self.threshold = threshold
        self.target = target
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __call__(self, image_array):
        """
        Args:
            image_array (np.ndarray): 3D DICOM数据数组 (depth, height, width)

        Returns:
            np.ndarray: 处理后的3D数组，非骨区域置0
        """

        if self.target == 'higher':
            # 创建骨组织掩码
            mask = image_array >= self.threshold
        else:
            # 创建掩码：骨区域为True（高于阈值）
            mask = image_array <= self.threshold
        # 将骨区域设为最低值（0），保留非骨区域
        processed_image = image_array * mask
        # 截断上下限，确保不会有CT值上下限不同导致0-255映射不一致。
        windowed = np.clip(processed_image, self.hu_min, self.hu_max)
        #  线性映射到 0-255
        normalized = (windowed - self.hu_min) / (self.hu_max - self.hu_min + 1e-6) * 255

        return normalized


# -------- albumentations自定义变换函数 --------
def fix_data_format(x, **kwargs):
    return np.array(x).squeeze().astype(np.float32)

def ensure_channel_dim(x, **kwargs):
    return x if x.ndim == 3 else np.expand_dims(x, -1)

def to_chw(x, **kwargs):
    return np.transpose(x, (2, 0, 1))

def hu_to_01(hu, HU_MIN=-1000, HU_MAX=2000, **kwargs):
    """
    截断HU值上下限后再归一化。
    hu: numpy array, dtype=np.int16 or float32, shape (H,W) or (H,W,1) 带CT值信息的数组
    return: float32 (H,W,1) 已归一化到 [0,1]
    """
    hu = hu.astype(np.float32)
    hu_clip = np.clip(hu, HU_MIN, HU_MAX)
    x_01 = (hu_clip - HU_MIN) / (HU_MAX - HU_MIN)   # 0~1

    return x_01

def norm_to_hu(x_norm, HU_MIN=-1000, HU_MAX=2000):
    """
    将 [-1,1] 映射回 [HU_MIN, HU_MAX]
    同时支持 torch.Tensor 和 numpy.ndarray
    返回与输入同类型、同设备的张量/数组
    """
    if isinstance(x_norm, torch.Tensor):
        x_01 = (x_norm + 1) / 2
        hu = x_01 * (HU_MAX - HU_MIN) + HU_MIN
        return hu
    elif isinstance(x_norm, np.ndarray):
        x_01 = (x_norm + 1) / 2
        hu = x_01 * (HU_MAX - HU_MIN) + HU_MIN
        return hu.astype(np.float32)
    else:
        raise TypeError('x_norm must be torch.Tensor or np.ndarray')

def apply_window(hu, WW=100, WL=50):
    """
    窗宽窗位转换，同时支持 torch.Tensor 和 numpy.ndarray
    只做窗宽位约束并归一化。
    """
    window_min = WL - WW // 2
    window_max = WL + WW // 2

    if isinstance(hu, torch.Tensor):
        windowed = hu.clamp(window_min, window_max)
        png_01 = (windowed - window_min) / WW  # [0,1]
        png_01 = png_01.clamp(0, 1) # 确保值域在[0,1]
    elif isinstance(hu, np.ndarray):
        windowed = np.clip(hu, window_min, window_max)
        png_01 = np.clip((windowed - window_min) / WW, 0, 1)    # [0,1]
    else:
        raise TypeError('hu must be torch.Tensor or np.ndarray')
    return png_01
