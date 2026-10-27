# metrics/FID_KID_3Dto2D.py
"""
3-D nii → 2-D PNG（过滤无意义切片）→ torch_fidelity 计算 FID/KID
输出：
/model_sample/{savename}:
-/fake_png/{PID}_*.png 生成结果的轴向切片（已根据真实数据排除了无诊断意义的切片）。
-/real_png/{PID}_*.png 真实目标结果的轴向切片（已根据真实数据排除了无诊断意义的切片）。
/model_save/{savename}:
-metrics_FID_KID.csv FID、KID结果。
-FID_KID_used_images.csv 计算FID、KID结果用到的图像。

【新增功能】
支持 use_raw_data=True 模式，与 metricsMAE_RMSE_SSIM_PSNR_3D.py 保持一致：
- NCCT2CTA: 计算 CTA（目标域/real）vs NCCT（条件域/fake）的 FID/KID
- CTA2NCCT: 计算 NCCT（目标域/real）vs CTA（条件域/fake）的 FID/KID
此时不读取模型生成结果，直接对比原始条件域与目标域的分布差异。
结果保存到 /home/longkaixing/NCCT_CTA/model_sample/{direction} 下。
测试集患者列表通过 sampler_dir + fold 的 test_idx 筛选，与 3D 指标脚本完全一致。
"""
import os, glob, sys
from os.path import join
# 设置Python路径，确保能找到tbe模块
cann_python_path = '/home/HwHiAiUser/Ascend/ascend-toolkit/8.2.RC1/python/site-packages'
os.environ['PYTHONPATH'] = cann_python_path + (':' + os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '')
sys.path.insert(0, cann_python_path)  # 双保险
import torch
import torch_npu
torch.npu.set_device('npu:7')  # 显式设置设备，触发上下文初始化
from torch_npu.contrib import transfer_to_npu
torch.set_default_device('npu:7')
import cv2, numpy as np, nibabel as nib
from tqdm import tqdm
import torch_fidelity as tf
import pandas as pd
import datetime

import tempfile, time
import random, matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from util.HU_utils import hu2uint8

# ------------------------------------------------
# 1. 路径与参数
# ------------------------------------------------
data_dir = "/home/longkaixing/NCCT_CTA/data/3_ResampleVolume"  # 真值
external_data_dir = '/home/longkaixing/ctaSim/data/real_cta'  # 外部测试集原始数据

# 新增：方向参数，决定谁是 GT，谁是 Input
# "NCCT2CTA": GT=CTA, Input=NCCT (计算 CTA 分布与 NCCT 分布的距离)
# "CTA2NCCT": GT=NCCT, Input=CTA (计算 NCCT 分布与 CTA 分布的距离)
direction = "NCCT2CTA"

# 新增：是否使用原始数据对比模式
# False: 评估模型生成结果 vs 目标域（默认）
# True:  评估原始条件域 vs 目标域（NCCT 与 CTA 的固有分布差异）
use_raw_data = False
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/NCCT2CTA'   # todo
# ---------- 测试集划分参数（与 metricsMAE_RMSE_SSIM_PSNR_3D.py 完全一致） ----------
sampler_dir = '/home/longkaixing/NCCT_CTA/data/kfold_split_idx/260130_CTA2NCCT'
fold = 1

"校正前"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/CTA'
"Pix2Pix_3D"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260130_Pix2Pix_3D'    # patch_size=224
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260224_Pix2Pix_3D'     # patch_size=112

"EDM_3D"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260223_EDM_3D'  # patch_size=112
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260226_EDM_3D'    # patch_size=56
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260226_EDM_3D_regression'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260226_EDM_3D_reg_expA_4'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260226_EDM_3D_reg_expA_10'

# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260313_EDM_3D_freq'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260320_EDM_3D_decoupled'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260414_EDM_3D_decoupled'

"DDBM"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260430_DDBM_3D'

"RectifiedFlow"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260501_RectifiedFlow'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260501_RectifiedFlow_regression'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260501_RectifiedFlow_reg_expA_4'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260501_RectifiedFlow_reg_expA_10'

# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260531_RectifiedFlow_iREPA'.
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260614_RectifiedFlow_DeEq'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260615_RectifiedFlow_DeEq'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260626_RectifiedFlow_DeEq'

"MeanFlows"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260512_MeanFlows_3D'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260512_MeanFlows_3D_regression'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260512_MeanFlows_3D_reg_expA_4'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260512_MeanFlows_3D_reg_expA_10'

# ============== NCCT → CTA ==================
"Pix2Pix"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260828_Pix2Pix'

"NeMAR"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260906_NeMAR'

"RegGAN"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260903_RegGAN'

"DefEquiv"
#sample_dir = ''

"RegConDID"
# sample_dir = ''

"SMILE"
sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260928_SMILE'

"RectifiedFlow_DeEq"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260717_RectifiedFlow_DeEq'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260717_RectifiedFlow_DeEq_regression'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260717_RectifiedFlow_DeEq_reg_expA_4'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260717_RectifiedFlow_DeEq_reg_expA_10'
"RectifiedFlow"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260808_RectifiedFlow'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260808_RectifiedFlow_regression'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260808_RectifiedFlow_reg_expA_4'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260808_RectifiedFlow_reg_expA_10'
"RectifiedFlow_iREPA"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260818_RectifiedFlow_iREPA'

# 注意：pred_root 在 use_raw_data=True 时不会被使用，但保留定义以兼容模型评估模式
# pred_root = '/home/longkaixing/NCCT_CTA/data/3_ResampleVolume'    # todo CTA vs NCCT时
pred_root = join(sample_dir, '3D_nii')

HU_WW, HU_WL = 350, 40  # 窗宽窗位
SIZE = 448  # 输出分辨率
SLICE_STEP = 1  # 每 N 层抽一张
FG_THRESH = 0.05  # 前景占比<5% 丢弃
FG_HU = -900    # 前景阈值下限
TISSUE_HU = [-200, 200] # 软组织阈值范围

# ------------------------------------------------
# 2. 工具函数
# ------------------------------------------------
def rotate_cw(img):
    """顺时针旋转图像90度（用于摆正轴向切片）"""
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def save_paired_slices(base_nii, real_nii, fake_nii, out_real_dir, out_fake_dir, prefix):
    """
    在 base_nii 上过滤，但 real_nii 和 fake_nii 同一切面都保存
    返回：保存张数（real 与 fake 始终相同）
    """
    base_vol = nib.load(base_nii).get_fdata(dtype=np.float32)
    real_vol = nib.load(real_nii).get_fdata(dtype=np.float32)
    fake_vol = nib.load(fake_nii).get_fdata(dtype=np.float32)
    fg_mask = base_vol > FG_HU  # ① 以 base 为准过滤
    D = base_vol.shape[2]
    cnt = 0
    for z in range(0, D, SLICE_STEP):
        fg_slc = fg_mask[:, :, z]
        if fg_slc.sum() / fg_slc.size < FG_THRESH:  # ② 只在 base 上过滤
            continue
        # ③ 两张都存
        real_png = hu2uint8(real_vol[:,:,z], HU_WW, HU_WL)
        fake_png  = hu2uint8(fake_vol[:,:,z], HU_WW, HU_WL)
        real_png = cv2.resize(real_png, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        fake_png = cv2.resize(fake_png, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        real_rgb = cv2.cvtColor(real_png, cv2.COLOR_GRAY2RGB)
        fake_rgb = cv2.cvtColor(fake_png, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(join(out_real_dir, f"{prefix}_{z:03d}.png"), real_rgb)
        cv2.imwrite(join(out_fake_dir, f"{prefix}_{z:03d}.png"), fake_rgb)
        cnt += 1
    return cnt


def save_external_pair(cta_nii, pred_nii, out_cta_dir, out_fake_dir, prefix):
    """
    外部测试集：同时保存输入和输出
    返回：保存的切片数量
    """
    cta_vol = nib.load(cta_nii).get_fdata(dtype=np.float32)
    pred_vol = nib.load(pred_nii).get_fdata(dtype=np.float32)

    # 基于输出计算前景掩码（或基于输入，保持一致即可）
    fg_mask = pred_vol > FG_HU
    D = pred_vol.shape[2]
    cnt = 0

    for z in range(0, D, SLICE_STEP):
        fg_slc = fg_mask[:, :, z]
        if fg_slc.sum() / fg_slc.size < FG_THRESH:
            continue

        # 保存输入
        cta_png = hu2uint8(cta_vol[:, :, z], HU_WW, HU_WL)
        cta_png = cv2.resize(cta_png, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        cta_rgb = cv2.cvtColor(cta_png, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(join(out_cta_dir, f"{prefix}_{z:03d}.png"), cta_rgb)

        # 保存输出
        pred_png = hu2uint8(pred_vol[:, :, z], HU_WW, HU_WL)
        pred_png = cv2.resize(pred_png, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        pred_rgb = cv2.cvtColor(pred_png, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(join(out_fake_dir, f"{prefix}_{z:03d}.png"), pred_rgb)

        cnt += 1
    return cnt


def robust_imwrite(img, path, max_retry=5, sleep=0.1):
    """原子写 + 重试，任何失败都抛错停脚本"""
    for i in range(max_retry):
        try:
            fd, tmp = tempfile.mkstemp(suffix='.png', dir=os.path.dirname(path))
            os.close(fd)
            success = cv2.imwrite(tmp, img)
            if not success:
                raise RuntimeError(f"cv2.imwrite returned False on {tmp}")
            os.replace(tmp, path)  # 原子覆盖
            return  # 成功立即返回
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if i == max_retry - 1:  # 最后一次也失败
                raise e
            time.sleep(sleep)  # 稍等再重试


def _repair_one_case(pid, out_real_dir, out_fake_dir, use_raw=False, direction="NCCT2CTA"):
    """
    仅针对当前患者：
    1. 列出真实目录里已存在的该患者 PNG；
    2. 根据「以 base 为基准的过滤逻辑」算出本该有的文件名；
    3. 缺失哪张补哪张（原子写 + 重试）。
    """
    # 根据 use_raw_data 和 direction 确定 base / real / fake 文件路径
    if use_raw:
        if direction == "NCCT2CTA":
            base_nii = join(data_dir, pid, "CTA.nii.gz")
            real_nii = join(data_dir, pid, "CTA.nii.gz")
            fake_nii = join(data_dir, pid, "NCCT.nii.gz")
        else:  # CTA2NCCT
            base_nii = join(data_dir, pid, "NCCT.nii.gz")
            real_nii = join(data_dir, pid, "NCCT.nii.gz")
            fake_nii = join(data_dir, pid, "CTA.nii.gz")
    else:
        # 模型评估模式：real=目标域真实值，fake=模型预测，base以目标域为准
        if direction == "NCCT2CTA":
            target_mod = "CTA"
        else:  # CTA2NCCT
            target_mod = "NCCT"
        base_nii = join(data_dir, pid, f"{target_mod}.nii.gz")
        real_nii = join(data_dir, pid, f"{target_mod}.nii.gz")  # GT
        fake_nii = join(pred_root, f"{pid}_whole.nii.gz")      # 预测

    # 1. 当前患者已存在的文件（只扫 pid 前缀）
    real_set = set(os.path.basename(f) for f in
                   glob.glob(join(out_real_dir, f"{pid}_*.png")))
    fake_set = set(os.path.basename(f) for f in
                   glob.glob(join(out_fake_dir, f"{pid}_*.png")))

    # 2. 重新计算「本该有的文件名集合」（以 base 为基准）
    base_vol = nib.load(base_nii).get_fdata(dtype=np.float32)
    fg_mask   = base_vol > FG_HU
    common_base = set(f"{pid}_{z:03d}.png"
                      for z in range(0, base_vol.shape[2], SLICE_STEP)
                      if (fg_mask[:, :, z].sum() / fg_mask[:, :, z].size) >= FG_THRESH)

    missing_real = common_base - real_set
    missing_fake = common_base - fake_set
    if not missing_real and not missing_fake:
        return  # 无缺，直接返回

    # 3. 按需补写
    real_vol = nib.load(real_nii).get_fdata(dtype=np.float32)
    fake_vol = nib.load(fake_nii).get_fdata(dtype=np.float32)
    for z in range(0, base_vol.shape[2], SLICE_STEP):
        fname = f"{pid}_{z:03d}.png"
        if fname not in missing_real and fname not in missing_fake:
            continue  # 两张都存在，跳过
        fg_slc = fg_mask[:, :, z]
        if fg_slc.sum() / fg_slc.size < FG_THRESH:
            continue  # 过滤条件不变
        real_png = hu2uint8(real_vol[:, :, z], HU_WW, HU_WL)
        fake_png  = hu2uint8(fake_vol[:, :, z], HU_WW, HU_WL)
        real_png = cv2.resize(real_png, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        fake_png  = cv2.resize(fake_png,  (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        real_rgb = cv2.cvtColor(real_png, cv2.COLOR_GRAY2RGB)
        fake_rgb  = cv2.cvtColor(fake_png,  cv2.COLOR_GRAY2RGB)

        if fname in missing_real:
            robust_imwrite(real_rgb, join(out_real_dir, fname))
        if fname in missing_fake:
            robust_imwrite(fake_rgb,  join(out_fake_dir, fname))


def compute_metrics(y_true, y_pred):
    """y_true, y_pred 已经是 1-D 向量，HU值"""
    # PSNR
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = psnr(y_true, y_pred, data_range=3000)    # HU 动态范围 [-1000, 2000]

    # SSIM
    global_min = min(y_true.min(), y_pred.min())
    global_max = max(y_true.max(), y_pred.max())
    y_true_n = (y_true - global_min) / (global_max - global_min + 1e-9)
    y_pred_n = (y_pred - global_min) / (global_max - global_min + 1e-9)
    ssim_val = ssim(y_true_n, y_pred_n,
                    data_range=1.0,
                    win_size=min(7, len(y_true) // 2 * 2 + 1))

    return None, None, ssim_val, psnr_val  # 返回4个值保持接口一致


# ------------------------------------------------
# 3. 主流程
# ------------------------------------------------
assert direction in ("CTA2NCCT", "NCCT2CTA"), f"direction 必须为 'CTA2NCCT' 或 'NCCT2CTA'，当前为 '{direction}'"

# ================================================================
# 根据 use_raw_data 确定 sample_dir、输出目录、患者列表
# ================================================================
if use_raw_data:
    # 原始数据对比模式：结果保存到基线目录，如 .../model_sample/NCCT2CTA
    sample_dir = f'/home/longkaixing/NCCT_CTA/model_sample/{direction}'
    os.makedirs(sample_dir, exist_ok=True)

    # 加载测试集索引（与 metricsMAE_RMSE_SSIM_PSNR_3D.py 完全一致）
    test_index_path = join(sampler_dir, f'test_idx_k={fold}.npy')
    if not os.path.exists(test_index_path):
        raise FileNotFoundError(f"测试集索引文件未找到: {test_index_path}")
    test_index = np.load(test_index_path)

    # 构建完整患者列表（与采样脚本保持完全一致）
    all_valid_pids = []
    for pid in sorted(os.listdir(data_dir)):
        if os.path.isdir(join(data_dir, pid)):
            all_valid_pids.append(pid)

    # 根据索引筛选出最终的测试集 pid 列表
    pids = [all_valid_pids[i] for i in test_index]

    # 输出目录基于新的 sample_dir
    out_real = join(sample_dir, 'real_png')
    out_fake = join(sample_dir, 'fake_png')
    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)
else:
    # 模型评估模式：从 pred_root 获取患者列表
    pred_files = sorted([f for f in os.listdir(pred_root) if f.endswith("_whole.nii.gz")])
    pids = [f.replace("_whole.nii.gz", "") for f in pred_files]

    # 输出目录基于文件开头定义的 sample_dir
    out_real = join(sample_dir, 'real_png')
    out_fake = join(sample_dir, 'fake_png')
    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)

is_external = 'RealData' in sample_dir

# 外部测试集不支持 use_raw_data（无 NCCT/CTA 配对）
if is_external and use_raw_data:
    print("[WARN] 外部测试集模式不支持 use_raw_data，已自动设为 False")
    use_raw_data = False
    # 重新获取 pids（从 pred_root）
    pred_files = sorted([f for f in os.listdir(pred_root) if f.endswith("_whole.nii.gz")])
    pids = [f.replace("_whole.nii.gz", "") for f in pred_files]

# ------------------------------------------------
# 4. 3D → 2D 切片 + 3D SSIM/PSNR 计算
# ------------------------------------------------
n_saved = 0

if is_external:
    # 外部测试模式：先保存所有内部参考切片作为 real，同时计算 3D SSIM/PSNR
    print("外部测试模式：准备参考分布并计算3D SSIM/PSNR...")
    internal_pids = [d for d in os.listdir(data_dir) if os.path.isdir(join(data_dir, d))]

    # 定义CTA图像保存目录
    real_cta_dir = join(sample_dir, 'real_cta_png')
    os.makedirs(real_cta_dir, exist_ok=True)

    # 4.1 保存内部参考分布的 PNG（用于 FID/KID）
    for ref_pid in tqdm(internal_pids, desc="参考分布 nii→PNG"):
        ncct_nii = join(data_dir, ref_pid, "ncct.nii.gz")
        if os.path.exists(ncct_nii):
            n_saved += save_paired_slices(
                ncct_nii, ncct_nii,  # real和fake都存ncct，但后面fake会被覆盖
                out_real, out_fake, f"{ref_pid}")

    # 清空 fake 目录，重新保存外部测试集的校正结果
    for f in os.listdir(out_fake):
        os.remove(join(out_fake, f))

    # 4.2 保存外部测试集的 PNG（用于 FID/KID）并计算 3D SSIM/PSNR
    records = []
    for pid in tqdm(pids, desc="外部测试集 nii→PNG & 3D metrics"):
        # 保存 PNG（用于 FID/KID）
        cta_nii = join(external_data_dir, pid, 'cta.nii.gz')
        pred_nii = join(pred_root, f"{pid}_whole.nii.gz")

        n_fake = save_external_pair(
            cta_nii,
            pred_nii,
            real_cta_dir,  # 保存输入
            out_fake,  # 保存输出
            f"{pid}")

        # 计算 3D SSIM/PSNR（基于 cta 自身的 mask）
        if not os.path.exists(cta_nii) or not os.path.exists(pred_nii):
            print(f"[Skip] {pid} 文件缺失，跳过3D metrics")
            continue

        cta_vol = nib.load(cta_nii).get_fdata(dtype=np.float32)
        pred_vol = nib.load(pred_nii).get_fdata(dtype=np.float32)

        # 确保尺寸一致
        if cta_vol.shape != pred_vol.shape:
            print(f"[Skip] {pid} 尺寸不匹配 {cta_vol.shape} vs {pred_vol.shape}")
            continue

        # 创建3D mask（基于 cta 自身的 HU 值，近似划分）
        tissue_mask = (cta_vol >= TISSUE_HU[0]) & (cta_vol <= TISSUE_HU[1])
        fg_mask = cta_vol > FG_HU

        # 提取1D向量
        cta_tissue = cta_vol[tissue_mask]
        pred_tissue = pred_vol[tissue_mask]
        cta_fg = cta_vol[fg_mask]
        pred_fg = pred_vol[fg_mask]

        # 计算指标（使用与3D版本完全相同的compute_metrics逻辑）
        if len(cta_tissue) > 0 and len(pred_tissue) > 0:
            _, _, ssim_tissue, psnr_tissue = compute_metrics(cta_tissue, pred_tissue)
        else:
            ssim_tissue, psnr_tissue = np.nan, np.nan

        if len(cta_fg) > 0 and len(pred_fg) > 0:
            _, _, ssim_fg, psnr_fg = compute_metrics(cta_fg, pred_fg)
        else:
            ssim_fg, psnr_fg = np.nan, np.nan

        records.append({
            "PID": pid,
            "tissue_SSIM": ssim_tissue,
            "tissue_PSNR": psnr_tissue,
            "FG_SSIM": ssim_fg,
            "FG_PSNR": psnr_fg
        })

    # 保存 3D SSIM/PSNR CSV（与3D_MAE_PSNR_SSIM_RMSE.py格式完全一致）
    df = pd.DataFrame(records)

    # 追加均值与标准差
    summary = pd.DataFrame({
        "PID": ["mean", "std"],
        **{col: [df[col].mean(), df[col].std()]
           for col in df.columns if col != "PID"}
    })

    df = pd.concat([df, summary], ignore_index=True)
    df.to_csv(join(sample_dir, "metrics_SSIM_PSNR.csv"), index=False)

    print(f"3D SSIM/PSNR 评估完成，结果已保存至 {join(sample_dir, 'metrics_SSIM_PSNR.csv')}")
    pd.set_option('display.max_columns', None)
    print(df.describe())

else:
    # 内部验证模式
    # 若切换了 use_raw_data，先清理旧 PNG 避免混合
    if use_raw_data:
        print(f"[INFO] use_raw_data=True，清理旧 PNG 并计算 {direction} 的原始分布差异...")
        for f in os.listdir(out_real):
            os.remove(join(out_real, f))
        for f in os.listdir(out_fake):
            os.remove(join(out_fake, f))

    for pid in tqdm(pids, desc="nii→PNG"):
        if use_raw_data:
            # 原始数据对比模式：条件域 vs 目标域
            if direction == "NCCT2CTA":
                # 目标域=CTA（real），条件域=NCCT（fake），过滤基准=CTA（与 MAE_RMSE 一致，基于 GT）
                base_nii = join(data_dir, pid, "CTA.nii.gz")
                real_nii = join(data_dir, pid, "CTA.nii.gz")
                fake_nii = join(data_dir, pid, "NCCT.nii.gz")
            else:  # CTA2NCCT
                # 目标域=NCCT（real），条件域=CTA（fake），过滤基准=NCCT（基于 GT）
                base_nii = join(data_dir, pid, "NCCT.nii.gz")
                real_nii = join(data_dir, pid, "NCCT.nii.gz")
                fake_nii = join(data_dir, pid, "CTA.nii.gz")
        else:
            # 模型评估模式：根据 direction 决定目标域（GT）和条件域（输入）
            if direction == "NCCT2CTA":
                target_mod = "CTA"
            else:  # CTA2NCCT
                target_mod = "NCCT"
            # 过滤基准使用目标域（GT），保证切片位置与 GT 一致
            base_nii = join(data_dir, pid, f"{target_mod}.nii.gz")
            real_nii = join(data_dir, pid, f"{target_mod}.nii.gz")   # GT
            fake_nii = join(pred_root, f"{pid}_whole.nii.gz")        # 模型预测

        n_saved += save_paired_slices(
            base_nii, real_nii, fake_nii,
            out_real, out_fake, f"{pid}")

    # 验证是否配对保存成功（检查最后一个 pid，保持原有逻辑）
    if pids:
        last_pid = pids[-1]
        r_cnt = len(glob.glob(join(out_real, f"{last_pid}_*.png")))
        f_cnt = len(glob.glob(join(out_fake, f"{last_pid}_*.png")))
        if r_cnt != f_cnt:
            print(f"[WARN] {last_pid}: real={r_cnt}  fake={f_cnt}")
            _repair_one_case(last_pid, out_real, out_fake, use_raw_data, direction)

n_real = n_fake = n_saved  # 保证 1:1

print(f"保存 PNG 完成：real={n_real} 张，fake={n_fake} 张")

# ------------------------------------------------
# 5. 计算 FID / KID
# ------------------------------------------------
metrics = tf.calculate_metrics(
    input1=out_real,
    input2=out_fake,
    cuda=True,  # 自动用 GPU
    isc=False,
    fid=True,
    kid=True,
    verbose=True,
    batch_size=50,
    num_workers=0,
)
print("FID :", metrics['frechet_inception_distance'])
print("KID:", metrics['kernel_inception_distance_mean'],
      "±", metrics['kernel_inception_distance_std'])

csv_metrics = join(sample_dir, "metrics_FID_KID.csv")
csv_img_path = join(sample_dir, "FID_KID_used_images.csv")
vis_name = "compare_grid.jpg"

# 6. 指标
df_metric = pd.DataFrame([{
    "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "direction": direction,
    "use_raw_data": use_raw_data,
    "n_real": n_real,
    "n_fake": n_fake,
    "FID": metrics['frechet_inception_distance'],
    "KID_mean": metrics['kernel_inception_distance_mean'],
    "KID_std": metrics['kernel_inception_distance_std'],
}])
df_metric.to_csv(csv_metrics, index=False)
print(f"FID & KID 已保存至 {csv_metrics}")

# 7. 重新列出「真正用于 fidelity 计算」的 PNG 绝对路径
real_paths = sorted([join(out_real, f) for f in os.listdir(out_real) if f.endswith('.png')])
fake_paths = sorted([join(out_fake, f) for f in os.listdir(out_fake) if f.endswith('.png')])

# 对齐数量（外部测试集可能数量不一致）
min_len = min(len(real_paths), len(fake_paths))
if len(real_paths) != len(fake_paths):
    print(f"[WARN] real/fake 数量不一致 ({len(real_paths)} vs {len(fake_paths)})，截断到 {min_len}")
    real_paths = real_paths[:min_len]
    fake_paths = fake_paths[:min_len]

print(f"对齐后数量：real={len(real_paths)}, fake={len(fake_paths)}")

df_img_path = pd.DataFrame({
    "real_png": real_paths,
    "fake_png": fake_paths
})
df_img_path.to_csv(csv_img_path, index=False)
print(f"计算所用图像路径已保存至 {csv_img_path}")

# ------------------------------------------------------------------
# 8. 可视化对比
# ------------------------------------------------------------------
n_show = 10  # 想看的组数
img_size = SIZE  # 统一缩放尺寸

if is_external:
    # 外部测试集：对比 cta vs corrected
    cta_paths_vis = sorted(
        [join(real_cta_dir, f) for f in os.listdir(real_cta_dir) if f.endswith('.png')])
    fake_paths_vis = sorted([join(out_fake, f) for f in os.listdir(out_fake) if f.endswith('.png')])

    # 确保文件名匹配
    cta_dict = {os.path.basename(p): p for p in cta_paths_vis}
    fake_dict = {os.path.basename(p): p for p in fake_paths_vis}
    common_files = sorted(set(cta_dict.keys()) & set(fake_dict.keys()))

    if len(common_files) == 0:
        print("[WARN] 无匹配的 cta/fake 图像用于可视化")
    else:
        # 随机选择 n_show 个
        n_show = min(n_show, len(common_files))
        idx_show = sorted(random.sample(range(len(common_files)), n_show))
        selected_files = [common_files[i] for i in idx_show]

        fig, axes = plt.subplots(nrows=len(selected_files), ncols=2, figsize=(6, 3 * len(selected_files)))
        if len(selected_files) == 1:
            axes = axes.reshape(1, -1)

        for row, fname in enumerate(selected_files):
            m_path = cta_dict[fname]
            f_path = fake_dict[fname]

            m_img = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
            f_img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
            m_img = rotate_cw(m_img)
            f_img = rotate_cw(f_img)

            if m_img is None or f_img is None:
                print(f"[WARN] 跳过对比 {fname} —— 图像读取失败")
                continue

            # 统一尺寸 & 转 RGB
            m_img = cv2.resize(m_img, (img_size, img_size))
            f_img = cv2.resize(f_img, (img_size, img_size))
            if m_img.ndim == 2:
                m_img = cv2.cvtColor(m_img, cv2.COLOR_GRAY2RGB)
                f_img = cv2.cvtColor(f_img, cv2.COLOR_GRAY2RGB)
            else:
                m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
                f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)

            axes[row, 0].imshow(m_img)
            axes[row, 0].set_title(f"cta  {fname}")
            axes[row, 0].axis('off')

            axes[row, 1].imshow(f_img)
            axes[row, 1].set_title(f"Corrected  {fname}")
            axes[row, 1].axis('off')

        plt.tight_layout()
        out_vis = join(sample_dir, vis_name)
        plt.savefig(out_vis, dpi=300, bbox_inches='tight')
        print(f"对比图已保存 → {out_vis}")
        plt.show()
else:
    # 内部验证模式
    real_paths = sorted([join(out_real, f) for f in os.listdir(out_real) if f.endswith('.png')])
    fake_paths = sorted([join(out_fake, f) for f in os.listdir(out_fake) if f.endswith('.png')])

    # 对齐数量
    min_len = min(len(real_paths), len(fake_paths))
    real_paths = real_paths[:min_len]
    fake_paths = fake_paths[:min_len]

    assert len(real_paths) == len(fake_paths), 'real/fake 数量不一致'

    idx_show = sorted(random.sample(range(len(real_paths)), min(n_show, len(real_paths))))

    if use_raw_data:
        # 原始对比模式：两列（Source vs Target）
        fig, axes = plt.subplots(nrows=len(idx_show), ncols=2, figsize=(6, 3 * len(idx_show)))
        if len(idx_show) == 1:
            axes = axes.reshape(1, -1)

        if direction == "NCCT2CTA":
            source_label = "Source (NCCT)"
            target_label = "Target (CTA)"
        else:  # CTA2NCCT
            source_label = "Source (CTA)"
            target_label = "Target (NCCT)"

        for row, idx in enumerate(idx_show):
            r_path = real_paths[idx]   # 这里real存储的是目标域（Target）
            f_path = fake_paths[idx]   # 这里fake存储的是条件域（Source）

            # 加载图像
            target_img = cv2.imread(r_path, cv2.IMREAD_UNCHANGED)
            source_img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
            target_img = rotate_cw(target_img)
            source_img = rotate_cw(source_img)
            if target_img is None or source_img is None:
                print(f"[WARN] 跳过对比 {os.path.basename(r_path)} —— 图像读取失败")
                continue

            target_img = cv2.resize(target_img, (img_size, img_size))
            source_img = cv2.resize(source_img, (img_size, img_size))
            if target_img.ndim == 2:
                target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)
                source_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)
            else:
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

            axes[row, 0].imshow(source_img)
            axes[row, 0].set_title(f"{source_label}  {os.path.basename(r_path)}")
            axes[row, 0].axis('off')

            axes[row, 1].imshow(target_img)
            axes[row, 1].set_title(f"{target_label}  {os.path.basename(r_path)}")
            axes[row, 1].axis('off')

        plt.tight_layout()
        out_vis = join(sample_dir, vis_name)
        plt.savefig(out_vis, dpi=300, bbox_inches='tight')
        print(f"对比图已保存 → {out_vis}")
        plt.show()

    else:
        # 模型评估模式：三列（Source, Pred, GT）
        # 需要动态加载 Source 图像
        # 先根据 direction 确定条件域模态
        if direction == "NCCT2CTA":
            source_mod = "NCCT"
            gt_mod = "CTA"
        else:  # CTA2NCCT
            source_mod = "CTA"
            gt_mod = "NCCT"

        # 预读所有文件的 pid 和 z
        # 从文件名解析 pid 和 z
        pid_z_pairs = []
        for fname in real_paths:
            base = os.path.basename(fname)  # e.g., "PID_023.png"
            pid, z_str = base.rsplit('_', 1)
            z = int(z_str.replace('.png', ''))
            pid_z_pairs.append((pid, z, fname))

        # 随机选择 n_show 个
        if len(pid_z_pairs) == 0:
            print("[WARN] 无图像用于可视化")
        else:
            selected = random.sample(pid_z_pairs, min(n_show, len(pid_z_pairs)))
            n_rows = len(selected)
            fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(9, 3 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for row, (pid, z, gt_path) in enumerate(selected):
                # 读取 GT（已存在）
                gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                gt_img = rotate_cw(gt_img)
                if gt_img is None:
                    print(f"[WARN] 跳过 {pid}_{z:03d} —— GT 读取失败")
                    continue

                # 读取 Pred（fake 目录下同名文件）
                pred_path = join(out_fake, f"{pid}_{z:03d}.png")
                pred_img = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
                pred_img = rotate_cw(pred_img)
                if pred_img is None:
                    print(f"[WARN] 跳过 {pid}_{z:03d} —— Pred 读取失败")
                    continue

                # 动态加载 Source
                source_nii_path = join(data_dir, pid, f"{source_mod}.nii.gz")
                if not os.path.exists(source_nii_path):
                    print(f"[WARN] 跳过 {pid}_{z:03d} —— Source nii 不存在")
                    continue
                source_vol = nib.load(source_nii_path).get_fdata(dtype=np.float32)
                # 检查 z 是否在范围内
                if z >= source_vol.shape[2]:
                    print(f"[WARN] 跳过 {pid}_{z:03d} —— z 超出 Source 范围")
                    continue
                source_slice = source_vol[:, :, z]
                source_png = hu2uint8(source_slice, HU_WW, HU_WL)
                source_png = cv2.resize(source_png, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
                source_img = cv2.cvtColor(source_png, cv2.COLOR_GRAY2RGB)
                source_img = rotate_cw(source_img)

                # 统一尺寸与颜色空间
                gt_img = cv2.resize(gt_img, (img_size, img_size))
                pred_img = cv2.resize(pred_img, (img_size, img_size))
                if gt_img.ndim == 2:
                    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
                else:
                    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                if pred_img.ndim == 2:
                    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)
                else:
                    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

                # 显示 PID 和切片编号（与文件名一致）
                label_suffix = f"{pid}_{z:03d}"

                # 绘制三列
                axes[row, 0].imshow(source_img)
                axes[row, 0].set_title(f"Source ({source_mod})  {label_suffix}")
                axes[row, 0].axis('off')

                axes[row, 1].imshow(pred_img)
                axes[row, 1].set_title(f"Pred ({gt_mod})  {label_suffix}")
                axes[row, 1].axis('off')

                axes[row, 2].imshow(gt_img)
                axes[row, 2].set_title(f"GT ({gt_mod})  {label_suffix}")
                axes[row, 2].axis('off')

            plt.tight_layout()
            out_vis = join(sample_dir, vis_name)
            plt.savefig(out_vis, dpi=300, bbox_inches='tight')
            print(f"对比图已保存 → {out_vis}")
            plt.show()