# util/train_utils.py
"""
训练实用工具：配置保存、配置加载、断点保存、断点恢复、损失函数、损失写入csv管理类
"""
import os, csv, yaml, shutil, datetime, logging, glob
from os.path import join
from collections import OrderedDict
from easydict import EasyDict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch



FIRST_PRINT = True # 只打印一次的flag

# -------- 加载配置文件 --------
def load_config(config_path):
    """
    载入配置文件，返回字典形式。
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)

# -------- 保存配置文件 --------
def save_config(config_path, save_dir, savename='config'):
    """
    将配置文件保存为 YAML 格式，文件名包含当前时间。
    :param config_path: 配置字典的路径。
    :param save_dir: 保存目录。
    :param savename: 文件名前缀，默认为 "config"。
    """
    # 获取当前时间并格式化为字符串
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 构造文件名
    filename = f"{savename}_{current_time}.yaml"
    # 构造完整的保存路径
    save_path = os.path.join(save_dir, filename)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(config_path, save_path)

    return save_path


class EpochCSVLogger:
    """
    极简Epoch级CSV记录器，支持断点恢复覆盖
    """
    def __init__(self, save_path, filename, resume=False):
        save_dir = join(save_path, '结果csv')
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = join(save_dir, filename)
        self.resume = resume and os.path.exists(self.filepath)

        # 存储所有数据：OrderedDict保证epoch顺序
        self.data = OrderedDict()

        # 恢复模式：加载已有数据
        if self.resume:
            self._load()

    def _load(self):
        """加载已有CSV数据到内存"""
        with open(self.filepath, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row['epoch'])
                # 删除epoch列，其余作为该epoch的数据
                row.pop('epoch')
                self.data[epoch] = row

    def add_epoch(self, epoch, **kwargs):
        """
        记录/覆盖一个epoch的数据

        Args:
            epoch: 轮次（主键）
            **kwargs: 任意列名和值
        """
        # 字符串化所有值，保持原样
        row_data = {k: str(v) for k, v in kwargs.items()}

        if epoch in self.data:
            # 覆盖：更新已有epoch
            self.data[epoch].update(row_data)
        else:
            # 新增：创建新epoch
            self.data[epoch] = row_data

        # 立即写入磁盘（防崩溃丢失）
        self._flush()

    def _flush(self):
        """刷新到文件"""
        if not self.data:
            return

        # 收集所有列名（按首次出现的epoch顺序）
        all_keys = set()
        for row in self.data.values():
            all_keys.update(row.keys())
        fieldnames = ['epoch'] + sorted(all_keys)  # epoch第一，其余按字母序

        # 写入文件
        with open(self.filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for epoch, row in self.data.items():
                row_with_epoch = {'epoch': epoch}
                row_with_epoch.update(row)
                writer.writerow(row_with_epoch)


def add_column_titles(grid_np, titles, fontsize=None, font_color=(0, 255, 255)):
    """
    grid_tensor: [3, H, W]  已拼好的大图
    titles: List[str]  长度=列数
    返回: PIL.Image  带文字标注
    """
    pil = Image.fromarray(grid_np)   # 已经是 (H,W,3) 的 uint8
    draw = ImageDraw.Draw(pil)
    font = fontsize if fontsize is None else ImageFont.load_default()

    ncol = len(titles)
    w, h = pil.size          # (W, H)
    col_w = w // ncol
    y = 5                    # 顶部留边
    for idx, title in enumerate(titles):
        x = idx * col_w
        draw.text((x, y), title, font=font, fill=font_color)

    return pil


def plot_intensity_histogram(cond_HU, fake_HU, real_HU, vis_dir, epoch,
                              hu_min=-1000, hu_max=1000, num_bins=500,
                              smooth_window=5):
    """
    绘制CTA、生成NCCT、真实NCCT的CT值强度分布直方图（连续折线形式）

    Args:
        cond_HU: CTA输入 (B,1,H,W,D)
        fake_HU: 生成的NCCT (B,1,H,W,D)
        real_HU: 真实的NCCT (B,1,H,W,D)
        vis_dir: 可视化输出目录
        epoch: 当前轮次
        hu_min, hu_max: HU值范围
        num_bins: 直方图bin数量
        smooth_window: 平滑窗口大小（奇数）
    """

    # 配置参数（参考 Histogram.py）
    SMOOTHING = True

    # 颜色配置（参考 Histogram.py 风格）
    COLORS = {
        'CTA': '#A5A5A5',  # 灰色 - 输入
        'Real_NCCT': '#2ecc71',  # 绿色 - 真实目标
        'Fake_NCCT': '#7030A0'  # 紫色 - 生成结果
    }

    # 获取中心切片索引
    sl = cond_HU.size(-1) // 2

    # 准备数据：分别提取切片和体积
    # 切片数据 (取每个样本的中心切片)
    cond_slice = cond_HU[:, 0, :, :, sl].numpy().flatten()  # (B*H*W,)
    fake_slice = fake_HU[:, 0, :, :, sl].numpy().flatten()
    real_slice = real_HU[:, 0, :, :, sl].numpy().flatten()

    # 体积数据 (整个subvolume)
    cond_vol = cond_HU[:, 0].numpy().flatten()
    fake_vol = fake_HU[:, 0].numpy().flatten()
    real_vol = real_HU[:, 0].numpy().flatten()

    # 过滤HU范围并计算直方图
    def _compute_histogram(data, hu_min, hu_max, num_bins):
        """计算直方图，返回bin中心和概率密度"""
        # 过滤范围
        mask = (data >= hu_min) & (data <= hu_max)
        filtered = data[mask]

        if len(filtered) == 0:
            return None, None, 0

        # 计算直方图
        bins = np.linspace(hu_min, hu_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        counts, _ = np.histogram(filtered, bins=bins)
        density = counts / (counts.sum() * bin_width) if counts.sum() > 0 else counts

        return bin_centers, density, len(filtered)

    def _smooth_curve(x, y, window_size):
        """移动平均平滑"""
        if window_size < 3 or window_size % 2 == 0:
            window_size = max(3, window_size + 1)
        half_window = window_size // 2
        y_smooth = np.copy(y).astype(float)
        for i in range(half_window, len(y) - half_window):
            y_smooth[i] = np.mean(y[i - half_window:i + half_window + 1])
        return x, y_smooth

    # 数据配置列表
    datasets = [
        ('Slice', cond_slice, fake_slice, real_slice, f'epoch{epoch:04d}_Histogram_slice.png'),
        ('Volume', cond_vol, fake_vol, real_vol, f'epoch{epoch:04d}_Histogram_volume.png')
    ]

    for data_type, cond_data, fake_data, real_data, filename in datasets:
        fig, ax = plt.subplots(figsize=(12, 7))

        stats_info = []

        # 绘制三条曲线
        for name, data, color in [
            ('CTA (Input)', cond_data, COLORS['CTA']),
            ('Real NCCT (Target)', real_data, COLORS['Real_NCCT']),
            ('Generated NCCT', fake_data, COLORS['Fake_NCCT'])
        ]:
            bin_centers, density, pixel_count = _compute_histogram(
                data, hu_min, hu_max, num_bins
            )

            if bin_centers is None:
                continue

            # 平滑处理
            if SMOOTHING:
                bin_centers, density = _smooth_curve(bin_centers, density, smooth_window)

            # 绘制折线
            ax.plot(bin_centers, density,
                    label=name,
                    color=color,
                    linewidth=2.5,
                    linestyle='-',
                    alpha=0.9)

            # 记录统计信息
            stats_info.append({
                'name': name,
                'mean': np.mean(data),
                'std': np.std(data),
                'count': pixel_count
            })

        # 添加参考线
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='Water (0 HU)')
        ax.axvline(x=50, color='lightcoral', linestyle=':', linewidth=2, alpha=0.8, label='Vessel (~50 HU)')

        # 设置图形属性
        ax.set_xlabel('HU Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title(f'CT Intensity Distribution - {data_type} Level\n'
                     f'Epoch {epoch} (HU: [{hu_min}, {hu_max}])',
                     fontsize=13, fontweight='bold', pad=15)

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([hu_min, hu_max])
        ax.set_ylim(bottom=0)

        # 添加统计信息文本框
        stats_text = 'Statistics:\n'
        for s in stats_info:
            stats_text += f'{s["name"].split()[0]}: μ={s["mean"]:.1f}, σ={s["std"]:.1f}\n'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # 保存
        save_path = join(vis_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # print(f'[Histogram] Saved {data_type} distribution plot: {save_path}')


def save_ckpt(save_dir, epoch, global_steps, model, opt, sch, k,
              scaler=None, DataParallel=None):
    """
    保存模型 checkpoint（仅 .pth 格式，避免 pickle 兼容性问题）。
    Args:
        save_dir: 保存目录
        epoch: 当前轮次
        global_steps: 全局步数
        model: 模型实例
        opt: 优化器实例
        sch: 学习率调度器实例
        k: 当前折数
        scaler: 混合精度训练的 GradScaler（可选）
        DataParallel: 是否使用了 nn.DataParallel
    Returns:
        str: 保存的文件路径
    """
    os.makedirs(save_dir, exist_ok=True)

    # 获取模型状态字典（处理 DataParallel 包装）
    if DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    # 构建 checkpoint（只包含基础数据类型）
    checkpoint = {
        'epoch': epoch,
        'global_steps': global_steps,
        'state_dict': state_dict,
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': sch.state_dict() if sch else None,
        'arch': model.__class__.__name__,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    }

    # 可选添加混合精度训练的 scaler
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # 保存
    save_path = os.path.join(save_dir, f'last_model_k={k}.pth')
    torch.save(checkpoint, save_path)

    return save_path

def load_ckpt(ckpt_dir, model, opt, sch, device, k):
    """
    Args:
        ckpt_dir: 保存点的目录。
        model: 要训练的模型。
        opt: 训练用的优化器。
        sch: 学习率调度其。
        device: 要加载到的显卡。
        k: 第k折。
    Returns: 当前训练开始的轮次，总的迭代步数。
    """
    start_epoch, global_steps = 1, 1
    # 检查目录是否存在
    if not os.path.exists(ckpt_dir):
        logging.info(f"\nCheckpoint directory '{ckpt_dir}' does not exist yet.")
        return start_epoch, global_steps

    # 查找最新的检查点文件
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, f'last_model_k={k}.pth'))
    if not checkpoint_files:
        logging.info(f"No checkpoint files found in '{ckpt_dir}'. Starting from epoch 1.")
        return start_epoch, global_steps

    # 加载最新的检查点文件
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint, map_location=device)

    # 恢复模型和优化器的状态，以及总迭代步数
    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    sch.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    global_steps = checkpoint['global_steps']

    logging.info(f'Checkpoint loaded from {latest_checkpoint} at epoch {start_epoch}')
    logging.info(f'Training from epoch {start_epoch}')  # 将信息记录到日志文件中

    return start_epoch, global_steps
