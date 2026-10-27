# data/Dataset/NCCT2CTA_dataset.py
import os, random
from os.path import join
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from data.preprocess.data_utils import (
    clip_intensity, normalize_intensity,
    crop_or_pad_volume, translate_image,
    define_patch_origins, sample_random_patches
)


class NCCT2CTA(Dataset):
    """
    NCCT到CTA转换的配对Patch数据集

    数据组织要求：
    data_dir/
        patient_001/
            ncct.nii.gz   # NCCT图像（源域，平扫）
            cta.nii.gz    # CTA图像（目标域，含造影剂）
        patient_002/
            ...

    训练目标：给定NCCT Patch，生成对应的CTA Patch
    """
    def __init__(self,
                 data_dir: str,
                 sampler: Optional[List[int]],
                 vol_size: Tuple[int, int, int],  # (H, W, D)
                 slice_start: Union[int, List[int]],  # 起始切片或范围
                 slice_count: int,  # 采样切片数
                 patch_size: int,
                 patch_stride: int,
                 grid_patch_num: int,  # 网格采样Patch数
                 random_patch_num: int,  # 随机采样Patch数
                 hu_range: Tuple[int, int],  # (HU_MIN, HU_MAX)
                 norm_factor: Union[int, str],
                 shuffle_patches: bool = False,
                 augment: bool = False,
                 augment_prob: float = 0.5,
                 ):
        """
        Args:
            data_dir: 数据根目录
            sampler: 病人索引采样器（用于划分train/val）
            vol_size: 目标体积大小 (H, W, D)
            slice_start: 起始切片索引或随机范围 [min, max]
            slice_count: 沿Z轴采样的切片数量
            patch_size: Patch的空间尺寸（正方形）
            patch_stride: Patch滑动步长
            grid_patch_num: 从规则网格中采样的Patch数量
            random_patch_num: 额外随机采样的Patch数量
            hu_range: HU值截断范围 (min, max)
            norm_factor: 归一化因子
            shuffle_patches: 是否打乱Patch顺序
            augment: 是否启用数据增强
        """
        super().__init__()
        self.data_dir = data_dir
        self.sampler = sampler
        self.vol_size = vol_size
        self.slice_start = slice_start
        self.slice_count = slice_count
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_patch_num = grid_patch_num
        self.random_patch_num = random_patch_num
        self.hu_min, self.hu_max = hu_range
        self.norm_factor = norm_factor
        self.shuffle_patches = shuffle_patches
        self.augment = augment
        self.augment_prob = augment_prob

        # 缓存计算得到的Patch原点（网格部分）
        self.grid_origins, _ = define_patch_origins(
            vol_size, patch_size, patch_stride
        )
        assert grid_patch_num <= len(self.grid_origins)

        # 扫描数据目录：NCCT为源，CTA为目标
        self.patient_ids, self.ncct_files, self.cta_files = self.scan_dataset(self.data_dir)

        # 应用采样器划分数据
        if sampler is not None:
            self.cta_files = [self.cta_files[i] for i in sampler]
            self.ncct_files = [self.ncct_files[i] for i in sampler]
            self.patient_ids = [self.patient_ids[i] for i in sampler]

        self.num_patients = len(self.cta_files)
        if self.num_patients == 0:
            raise RuntimeError(
                f'在 {data_dir} 中未找到有效的NCCT/CTA配对数据'
                f'（期望结构：patient_id/ncct.nii.gz & patient_id/cta.nii.gz）'
            )

        # 每个病人的总Patch数
        self.patches_per_patient = grid_patch_num + random_patch_num

        # 生成索引数组 [(patient_idx, patch_idx), ...]
        self.index_map = self._build_index_map()

        # 运行时缓存
        self._reset_cache()

        # 当前病人的Patch原点列表（包含网格+随机）
        self.current_patch_origins: List[Tuple[int, int]] = []

    @staticmethod
    def scan_dataset(data_dir) -> Tuple[List[str], List[str], List[str]]:
        """
        扫描目录，返回病人ID和文件路径列表

        期望文件结构：
            patient_id/cta.nii.gz (源图像)
            patient_id/ncct.nii.gz  (目标图像)
        """
        ids, cta_list, ncct_list = [], [], []

        for pid in sorted(os.listdir(data_dir)):
            ncct_path = join(data_dir, pid, 'NCCT.nii.gz')
            cta_path = join(data_dir, pid, 'CTA.nii.gz')

            if os.path.isfile(ncct_path) and os.path.isfile(cta_path):
                ids.append(pid)
                ncct_list.append(ncct_path)
                cta_list.append(cta_path)
            else:
                print(f'[warning-NCCT2CTA_sacn_dataset] {ncct_path} or {cta_path} not exist!')

        return ids, ncct_list, cta_list

    def _build_index_map(self) -> List[List[int]]:
        """构建样本索引映射表"""
        indices = []
        patient_indices = np.arange(self.num_patients)

        for p_idx in patient_indices:
            if self.shuffle_patches:
                patch_order = np.random.permutation(self.patches_per_patient)
            else:
                patch_order = np.arange(self.patches_per_patient)

            for patch_idx in patch_order:
                indices.append([p_idx, patch_idx])

        return indices

    def _reset_cache(self):
        """重置数据缓存"""
        self.cache = {
            'patient_id': None,
            'ncct_file': None,  # 源文件路径 (NCCT)
            'cta_file': None,  # 目标文件路径 (CTA)
            'ncct_norm': None,  # 归一化后的NCCT图像
            'cta_norm': None,  # 归一化后的CTA图像
            'ncct_hu': None,  # 原始NCCT的HU值
            'cta_hu': None,  # 原始CTA的HU值
        }

    def _sample_patches_for_patient(self):
        """为当前病人采样Patch原点"""
        # 网格部分
        if self.shuffle_patches:
            grid_samples = random.sample(self.grid_origins, self.grid_patch_num)
        else:   # 固定部分确保覆盖全图，随机部分增加多样性/随机性
            grid_samples = self.grid_origins[:self.grid_patch_num]

        # 随机部分
        random_samples = sample_random_patches(
            self.grid_origins, self.random_patch_num
        )

        self.current_patch_origins = grid_samples + random_samples

    def _load_volume(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载NIfTI文件并进行预处理
        Returns:
            hu_img: 原始HU值图像
            norm_img: 归一化后的图像
        """
        hu_img = nib.load(filepath).get_fdata()

        processed = clip_intensity(hu_img, self.hu_min, self.hu_max)
        norm_img = normalize_intensity(
            processed, self.norm_factor,
            (self.hu_min, self.hu_max), (-1, 1)
        )

        return hu_img, norm_img

    def _get_slice_range(self, vol_depth: int) -> Tuple[int, int]:
        """确定Z轴采样范围"""
        if isinstance(self.slice_start, int):
            start = self.slice_start
        else:
            min_start, max_start = self.slice_start
            max_start = min(max_start, vol_depth - self.slice_count - 2)
            start = random.randint(min_start, max_start)

        return start, start + self.slice_count

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_idx, patch_idx = self.index_map[idx]
        patient_id = self.patient_ids[patient_idx]

        # 切换病人时更新缓存
        if patient_id != self.cache['patient_id']:
            self.cache['patient_id'] = patient_id

            # 加载NCCT（源域/条件）
            ncct_path = self.ncct_files[patient_idx]
            if ncct_path != self.cache['ncct_file']:
                ncct_hu, ncct_norm = self._load_volume(ncct_path)
                self.cache['ncct_file'] = ncct_path
                self.cache['ncct_hu'] = ncct_hu
                self.cache['ncct_norm'] = ncct_norm

            # 加载CTA（目标域）
            cta_path = self.cta_files[patient_idx]
            if cta_path != self.cache['cta_file']:
                cta_hu, cta_norm = self._load_volume(cta_path)
                self.cache['cta_file'] = cta_path
                self.cache['cta_hu'] = cta_hu
                self.cache['cta_norm'] = cta_norm

            # 为新病人重新采样Patch位置
            self._sample_patches_for_patient()

        # 获取当前Z轴范围
        depth = self.cache['ncct_norm'].shape[2]
        z_start, z_end = self._get_slice_range(depth)

        # 提取子体积
        ncct_vol = self.cache['ncct_norm'][:, :, z_start:z_end]
        cta_vol = self.cache['cta_norm'][:, :, z_start:z_end]

        # 裁剪/填充到统一大小
        target_3d = (*self.vol_size[:2], self.slice_count)
        ncct_vol = crop_or_pad_volume(ncct_vol, target_3d, np.min(ncct_vol))
        cta_vol = crop_or_pad_volume(cta_vol, target_3d, np.min(cta_vol))

        # 提取空间Patch
        ox, oy = self.current_patch_origins[patch_idx]
        ncct_patch = ncct_vol[ox:ox + self.patch_size, oy:oy + self.patch_size, :]
        cta_patch = cta_vol[ox:ox + self.patch_size, oy:oy + self.patch_size, :]

        # 数据增强（对CTA和NCCT应用相同空间变换，保持配准）
        if self.augment and random.random() < self.augment_prob:
            shift_x = random.randint(-10, 10)
            shift_y = random.randint(-10, 10)
            ncct_patch = translate_image(ncct_patch, (shift_x, shift_y, 0))
            cta_patch = translate_image(cta_patch, (shift_x, shift_y, 0))   # CT图像不易旋转，仅平移。

        # 转换为Tensor (C, H, W, D)
        ncct_tensor = torch.from_numpy(ncct_patch).unsqueeze(0).float()
        cta_tensor = torch.from_numpy(cta_patch).unsqueeze(0).float()

        return {
            'cond': ncct_tensor,    # 源图像 (输入)
            'target': cta_tensor,   # 目标图像 (Ground Truth)
            'z_start': z_start,
            'patient_id': patient_id,
            'patch_origin': (ox, oy),
        }

    def on_epoch_end(self):
        """每个epoch结束时调用，重置当前缓存和索引"""
        self._reset_cache()
        self.index_map = self._build_index_map()
        self.current_patch_origins = []
