# metrics/MAE_RMSE_SSIM_PSNR_3D.py
"""
按论文方式评估 foreground 的 MAE / RMSE / SSIM / PSNR
输出结果：
/model_save/{savename}：
-metrics_3D_nii.csv 三维数据的MAE、RMSE、SSIM、PSNR指标。
"""
import os
from os.path import join
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ---------- 路径填写 ----------
data_dir = "/home/longkaixing/NCCT_CTA/data/3_ResampleVolume"

# ---------- 方向参数 ----------
# "CTA2NCCT": 条件=CTA, 目标=NCCT → mask 基于 CTA, gt=CTA, pred=生成的NCCT
# "NCCT2CTA": 条件=NCCT, 目标=CTA → mask 基于 NCCT, gt=NCCT, pred=生成的CTA
direction = "NCCT2CTA"

# ---------- 评估模式 ----------
# False: 评估模型生成结果 vs ground truth（默认）
# True:  评估原始 NCCT vs CTA（条件 vs 目标之间的固有差异）
use_raw_ncct = False  # True
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/NCCT2CTA'
# ---------- 测试集划分参数 ----------
sampler = '/home/longkaixing/NCCT_CTA/data/kfold_split_idx/260130_CTA2NCCT' # 对应采样脚本中的 args.sampler
fold = 1 # 对应采样脚本中的 fold


"Pix2Pix_3D"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260130_Pix2Pix_3D'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260224_Pix2Pix_3D'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260329_Pix2Pix_3D'

"EDM_3D"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260223_EDM_3D'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260226_EDM_3D'
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

# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260531_RectifiedFlow_iREPA'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260614_RectifiedFlow_DeEq'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260615_RectifiedFlow_DeEq'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260626_RectifiedFlow_DeEq'

"MeanFlows"
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/260512_MeanFlows_3D'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260512_MeanFlows_3D_regression'
# sample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260512_MeanFlows_3D_reg_expA_4'
# ample_dir = '/home/longkaixing/NCCT_CTA/model_sample/YODA/260512_MeanFlows_3D_reg_expA_10'


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



pred_root = join(sample_dir, '3D_nii')
save_csv = join(sample_dir, 'metrics_3D_nii.csv')
# save_csv = '/home/longkaixing/NCCT_CTA/model_sample/metrics_3D_nii(ctaVSncct).csv'    # CTA结果保存路径
# ------------------------------
def load_nii(path):
    data = nib.load(path).get_fdata(dtype=np.float32)
    # CT 值截断到 [-1000, 1000]
    data = np.clip(data, -1000, 1000)
    return data

def compute_metrics(y_true, y_pred):
    """y_true, y_pred 已经是 1-D 向量"""
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # PSNR
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        psnr_val = float('inf')
    else:
        data_range = 2000          # HU 动态范围(依据训练设置background_cutoff: -1000 maximum_cutoff: 2000)
        psnr_val = 20 * np.log10(data_range) - 10 * np.log10(mse)

    # SSIM
    # 用两图共同的 global min/max 归一化(也可以用HU截断窗口归一化，这样具有物理意义)
    global_min = min(y_true.min(), y_pred.min())
    global_max = max(y_true.max(), y_pred.max())
    y_true_n = (y_true - global_min) / (global_max - global_min + 1e-9)
    y_pred_n = (y_pred - global_min) / (global_max - global_min + 1e-9)
    ssim_val = ssim(y_true_n, y_pred_n,
                    data_range=1.0,
                    win_size=min(7, len(y_true)//2*2+1))
    return mae, rmse, ssim_val, psnr_val

def main():
    assert direction in ("CTA2NCCT", "NCCT2CTA"), f"direction 必须为 'CTA2NCCT' 或 'NCCT2CTA'，当前为 '{direction}'"

    # 1. 加载测试集索引
    test_index_path = join(sampler, f'test_idx_k={fold}.npy')
    if not os.path.exists(test_index_path):
        raise FileNotFoundError(f"测试集索引文件未找到: {test_index_path}")
    test_index = np.load(test_index_path)

    # 2. 构建完整患者列表 (与采样脚本保持完全一致)
    all_valid_pids = []
    for pid in sorted(os.listdir(data_dir)):
        # 这里简单判断目录是否存在即可，采样脚本中会进一步检查 CTA.nii.gz
        if os.path.isdir(join(data_dir, pid)):
            all_valid_pids.append(pid)

    # 3. 根据索引筛选出最终的测试集 pid 列表
    pids = [all_valid_pids[i] for i in test_index]

    records = []

    for pid in tqdm(pids, desc="Evaluating"):
        if use_raw_ncct:
            # 原始数据对比模式：gt=目标域, pred=条件域
            if direction == "NCCT2CTA":
                gt_path   = join(data_dir, pid, "CTA.nii.gz")   # 目标域
                pred_path = join(data_dir, pid, "NCCT.nii.gz")  # 条件域
            else:  # CTA2NCCT
                gt_path   = join(data_dir, pid, "NCCT.nii.gz")  # 目标域
                pred_path = join(data_dir, pid, "CTA.nii.gz")   # 条件域
        else:
            # 模型评估模式：gt=目标域, pred=模型生成结果
            if direction == "CTA2NCCT":
                gt_path = join(data_dir, pid, "CTA.nii.gz")
            else:  # NCCT2CTA
                gt_path = join(data_dir, pid, "NCCT.nii.gz")
            pred_path = join(pred_root, f"{pid}_whole.nii.gz")

        if not os.path.isfile(gt_path) or not os.path.isfile(pred_path):
            print(f"[Skip] {pid} 文件缺失")
            continue

        gt   = load_nii(gt_path)
        pred = load_nii(pred_path)

        # 确保尺寸一致
        if gt.shape != pred.shape:
            print(f"[Skip] {pid} 尺寸不匹配 {gt.shape} vs {pred.shape}")
            continue

        # -----  mask（基于目标侧 ground truth）-----
        fg_mask = gt > -900                                # 前景：排除空气
        tissue_mask = (gt >= -200) & (gt <= 200)           # 软组织

        # -----  先索引成 1-D 向量 -----
        fg_true = gt[fg_mask]
        fg_pred = pred[fg_mask]
        tissue_true = gt[tissue_mask]
        tissue_pred = pred[tissue_mask]

        # -----  计算 -----
        fg_mae, fg_rmse, fg_ssim, fg_psnr = compute_metrics(fg_true, fg_pred)
        tissue_mae, tissue_rmse, tissue_ssim, tissue_psnr = compute_metrics(tissue_true, tissue_pred)

        records.append({
            "PID": pid,
            "FG_MAE": fg_mae,
            "FG_RMSE": fg_rmse,
            "FG_SSIM": fg_ssim,
            "FG_PSNR": fg_psnr,
            "tissue_MAE": tissue_mae,
            "tissue_RMSE": tissue_rmse,
            "tissue_SSIM": tissue_ssim,
            "tissue_PSNR": tissue_psnr
        })

    # ----- 保存 -----
    df = pd.DataFrame(records)
    df.to_csv(save_csv, index=False)

    # ----- 追加均值与标准差 -----
    summary = pd.DataFrame({
        "PID": ["mean", "std"],
        **{col: [df[col].mean(), df[col].std()]
           for col in df.columns if col != "PID"}
    })
    summary.to_csv(save_csv, index=False, mode="a", header=False)
    print(f"评估完成（direction={direction}, use_raw_ncct={use_raw_ncct}），结果（含均值±标准差）已保存至 {save_csv}")
    pd.set_option('display.max_columns', None)  # 不省略列
    print(df.describe())


if __name__ == "__main__":
    main()