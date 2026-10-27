# Models/FlowMatch/train_RectifiedFlow_DeEq.py
"""
训练Rectified Flow模型进行3D CTA -> NCCT的转换。
加入刚性形变的变形等变（Deformation Equivariant）设计.
输出结果：
/model_save/{savename}：
-/log/training_log.csv 运行日志
-/output_images/epoch*_sample0.nii.gz、epoch*.png 训练集生成三维结果及对应的中心轴向切片
-/tensorboard_logs/*.tfevents 写入tensorboard的损失、学习率等信息
-model_epoch*.pt 每轮模型权重、latest_model.pt 最新一轮模型权重
-*.yaml 当次运行配置
"""
import socket
socket.setdefaulttimeout(30)  # 30秒超时
import os, sys, logging, argparse, datetime, time, setproctitle
from os.path import join
# 设置
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'  # 只显示 Error
os.environ['ASCEND_LOG_DEVICE_FLUSH_TIMEOUT'] = '0'
# 设置Python路径，确保能找到tbe模块
cann_python_path = '/home/HwHiAiUser/Ascend/ascend-toolkit/8.2.RC1/python/site-packages'
os.environ['PYTHONPATH'] = cann_python_path + (':' + os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '')
sys.path.insert(0, cann_python_path)  # 双保险
# 设置解决精度模式和编译错误的环境变量
os.environ['ASCEND_LAUNCH_BLOCKING'] = '0'  # 仅在定位算子错误时临时使用。正常训练时务必将其设为 0 或移除。
os.environ['NPU_FUZZY_COMPILE'] = '0'  # 关闭模糊编译，强制精确匹配
os.environ['ASCEND_PRECISION_MODE'] = 'must_keep_origin_dtype'  # 强制保持原始精度

setproctitle.setproctitle("lkx RF_DE 7/20")  # todo
import numpy as np
from tensorboardX import SummaryWriter
from accelerate import Accelerator
import functions_collection as ff
import nibabel as nb
from itertools import cycle
from tqdm import tqdm
# util
from util.parser import logger_setup
import util.Data_processing as dp
from util.HU_utils import hu2rgb
from util.train_utils import load_config, save_config, add_column_titles, EpochCSVLogger
# Torch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch_npu
torch.autograd.set_detect_anomaly(True)  # 启用异常检测
# 数据模型
from data.Dataset.NCCT2CTA_dataset import NCCT2CTA
from data.preprocess.Transforms import apply_window, norm_to_hu
from Models.cDDPM import Unet3D  # 不使用SongUnet，因为要保持z轴上不下采样。
from Models.FlowMatch import RectifiedFlow_DeEq
from Models.cDDPM.util_cDDPM import make_grid, load_ckpt, divisible_by, save_ckpt
from ema_pytorch import EMA


def train_one_epoch(Aclrtor, rf_model, opt, pbar, global_steps, start_time, total_steps, writer):
    steps_in_epoch = 0
    epoch_loss = 0
    average_loss, epoch_loss_dict = [], {}
    rf_model.train()
    for _, batch in enumerate(pbar):
        opt.zero_grad()
        target = batch['target'].to(device)
        cond = batch['cond'].to(device)
        with Aclrtor.autocast():
            loss, loss_dict = rf_model(target, cond)

        steps_in_epoch += 1
        # 计算当前batch的损失
        batch_loss = loss.item()
        epoch_loss += batch_loss
        average_loss.append(loss.item())

        # 累加分项损失（用于epoch平均）
        for name, value in loss_dict.items():
            if name not in epoch_loss_dict:
                epoch_loss_dict[name] = 0.0
            epoch_loss_dict[name] += value

        Aclrtor.backward(loss)

        opt.step()

        # 每步记录到tensorboard（可选，避免太频繁）
        if writer is not None and Aclrtor.is_main_process and steps_in_epoch % 10 == 0:
            for name, value in loss_dict.items():
                writer.add_scalar(f'Loss_step/{name}', value, global_steps)

        # -------- 计算剩余时间 --------
        global_steps += 1
        total_elapsed = time.time() - start_time
        avg_time_per_step = total_elapsed / global_steps if global_steps > 0 else 0
        steps_left = total_steps - global_steps
        eta_seconds = int(steps_left * avg_time_per_step)
        eta_str = str(datetime.timedelta(seconds=eta_seconds))

        # 动态构建进度条显示
        postfix_dict = {"Loss": f"{batch_loss:.4f}", "ETA": eta_str}
        for name, value in loss_dict.items():
            if name != 'total' and value > 0:  # 跳过 total，显示各分项
                postfix_dict[name[:3]] = f"{value:.4f}"  # 缩写避免过长

        pbar.set_postfix(postfix_dict)
        pbar.update(1)

    # 计算epoch平均
    avg_loss_dict = {k: v / steps_in_epoch for k, v in epoch_loss_dict.items()}

    return average_loss, global_steps, avg_loss_dict  # 返回分项平均


def test_one_epoch(Aclrtor, rf_model, val_dl):
    rf_model.eval()
    val_loss_dict = {}  # 累加分项
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_dl, desc='Testing'):
            target = batch['target'].to(device)
            cond = batch['cond'].to(device)
            batch_size = target.size(0)

            with Aclrtor.autocast():
                loss, loss_dict = rf_model(target, cond)

            # 加权累加（按batch大小）
            for name, value in loss_dict.items():
                if name not in val_loss_dict:
                    val_loss_dict[name] = 0.0
                val_loss_dict[name] += value * batch_size

            total_samples += batch_size

    # 计算平均
    avg_val_loss_dict = {k: v / total_samples for k, v in val_loss_dict.items()}

    # 返回总损失列表（兼容旧接口）和分项字典
    return list(avg_val_loss_dict.values()), avg_val_loss_dict


# ----可视化 ----
@torch.no_grad()
def visualize_sample(model, ema, save_dir, vis_samples_num, batch_cond, real, epoch):
    model.eval()
    ema.ema_model.eval()
    device = next(model.parameters()).device

    vis_dir = join(save_dir, 'output_images')
    ff.make_folder([vis_dir])

    b = min(vis_samples_num, batch_cond.shape[0])
    cond = batch_cond[:b].to(device)  # (b, c, Z, H, W)
    real = real[:b].to(device)
    _, c, x, y, z = real.shape
    # 处理 batch 的 cond：从 (b, 1, Z, H, W) -> (b, 1, H, W, D)
    cond_np = cond.cpu().numpy()  # (b, 1, Z, H, W)
    cond_list = []
    for i in range(b):
        c_np = cond_np[i, 0]  # (H, W, Z)
        c_np = dp.crop_or_pad(c_np, [x, y, z], value=np.min(c_np))  # (H, W, D)
        cond_list.append(c_np)

    cond_processed = np.stack(cond_list, axis=0)  # (b, H, W, D)
    cond = torch.from_numpy(cond_processed).unsqueeze(1).to(device)  # (b, 1, H, W, D)

    # 【差异】RectifiedFlow采样返回(fake, nfe)
    fake, nfe = ema.ema_model.sample(condition=cond, batch_size=b)
    fake = fake.clamp(-1, 1).cpu()
    cond = cond.cpu()
    real = real.cpu()
    torch.cuda.empty_cache()

    # 映射回HU再保存.nii.gz
    fake_HU = norm_to_hu(fake, HU_MIN=args.hu_min, HU_MAX=args.hu_max)  # (B,1,H,W,D)
    cond_HU = norm_to_hu(cond, HU_MIN=args.hu_min, HU_MAX=args.hu_max)
    real_HU = norm_to_hu(real, HU_MIN=args.hu_min, HU_MAX=args.hu_max)

    # 取中间切片
    sl = fake_HU.size(-1) // 2  # D//2
    fake_sl = fake_HU[:, 0, :, :, sl]  # [B,H,W]
    cond_sl = cond_HU[:, 0, :, :, sl]
    real_sl = real_HU[:, 0, :, :, sl]

    # ---- 差异图（仍在 HU） ----
    diff_real = real_sl - cond_sl  # 真实差异 (B,H,W)
    diff_fake = fake_sl - cond_sl  # 生成差异
    diff_fake_R = fake_sl - real_sl  # 生成残差
    diff_2nd = diff_fake - diff_real  # 二阶残差

    # ---- 全局差异上下限 ----
    HU_DIFF_RNG = dict(diff_real=(-1000, 1000), diff_fake=(-1000, 1000),
                       diff_fake_R=(-1000, 1000), diff_2nd=(-1000, 1000), )

    # ---- 转换差异图为 RGB（seismic colormap）----
    # 窗宽窗位 -> png [0,1]
    cond_rgb = apply_window(cond_sl, args.WW, args.WL).unsqueeze(1).repeat(1, 3, 1, 1)  # (B,3, H,W)
    fake_rgb = apply_window(fake_sl, args.WW, args.WL).unsqueeze(1).repeat(1, 3, 1, 1)
    real_rgb = apply_window(real_sl, args.WW, args.WL).unsqueeze(1).repeat(1, 3, 1, 1)

    diff_real_rgb = hu2rgb(diff_real, *HU_DIFF_RNG['diff_real'])  # (B,3,H,W)
    diff_fake_rgb = hu2rgb(diff_fake, *HU_DIFF_RNG['diff_fake'])
    diff_fake_R_rgb = hu2rgb(diff_fake_R, *HU_DIFF_RNG['diff_fake_R'])
    diff_2nd_rgb = hu2rgb(diff_2nd, *HU_DIFF_RNG['diff_2nd'])

    def rot90_3ch(x):
        # x: (B,3,H,W)  ->  (B,3,W,H)
        return torch.rot90(x, k=1, dims=(2, 3))

    cond_rgb = rot90_3ch(cond_rgb)  # (B,3,W,H)
    fake_rgb = rot90_3ch(fake_rgb)
    real_rgb = rot90_3ch(real_rgb)
    diff_real_rgb = rot90_3ch(diff_real_rgb)
    diff_fake_rgb = rot90_3ch(diff_fake_rgb)
    diff_fake_R_rgb = rot90_3ch(diff_fake_R_rgb)
    diff_2nd_rgb = rot90_3ch(diff_2nd_rgb)

    # 按「列优先」顺序拼：先取每张图的第 i 个样本，再放下一张图
    # 得到 [B*7, 3, H, W]
    cols = torch.cat([torch.stack([cond_rgb[i], fake_rgb[i], real_rgb[i],
                                   diff_real_rgb[i], diff_fake_rgb[i], diff_fake_R_rgb[i], diff_2nd_rgb[i]], dim=0)
                      for i in range(b)], dim=0)  # (B*7, 3, H, W)
    grid = make_grid(cols, nrow=7, padding=2, pad_value=0, normalize=False)

    # 转成 (H, W, 3) uint8 给 PIL
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)

    titles = ["cond (cond)",  # 输入
              "fake (fake)",  # 生成
              "real (target)",  # 目标
              "(cond - real)",  # 输入与目标差异
              "(fake - real)",  # 生成与目标差异
              "(fake - cond)",  # 生成残差
              "(fake_R - real_R)", ]  # 二阶残差

    pil_img = add_column_titles(grid_np, titles, fontsize=2)
    pil_img.save(join(vis_dir, f"epoch{epoch:04d}.png"))

    for i in range(b):
        fakeB_np = fake_HU[i, 0].cpu().numpy()
        nb.save(nb.Nifti1Image(fakeB_np, np.eye(4)),
                os.path.join(vis_dir, f'epoch{epoch:04d}_sample{i}.nii.gz'))

    # 【差异】记录NFE到日志
    if hasattr(args, 'log_nfe') and args.log_nfe:
        logging.info(f"Epoch {epoch}: NFE = {nfe}")


# ======== 参数解析部分 ========
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='配置文件路径',   # todo
    # default='/home/longkaixing/NCCT_CTA/config/FlowMatch/260614_RectifiedFlow_DeEq.yaml')
    # default='/home/longkaixing/NCCT_CTA/config/FlowMatch/260615_RectifiedFlow_DeEq.yaml')
    # default='/home/longkaixing/NCCT_CTA/config/FlowMatch/260626_RectifiedFlow_DeEq.yaml')
    default='/home/longkaixing/NCCT_CTA/config/FlowMatch/260717_RectifiedFlow_DeEq.yaml')

args, _ = parser.parse_known_args()
config = load_config(args.cfg)
for k, v in config.items():
    setattr(args, k, v)

torch_npu.npu.set_device(args.device)
# 初始化分布式环境
Aclrtor = Accelerator(  # 分布式训练
    split_batches=False,  # 不切分batch
    gradient_accumulation_steps=1,
    mixed_precision="bf16",
)
device = Aclrtor.device
print(f"[!] Rank {Aclrtor.process_index} using logical device: {Aclrtor.device}")

# -------- 记录设置 --------
# 基础保存路径
args.save_path = join(args.save_dir, args.save_name)
os.makedirs(args.save_path, exist_ok=True)
logger_setup(args.save_path, 'training')  # 设置

# -------- 设置环境变量 --------
# 设置随机种子
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# -------- 数据预处理 --------
args.t = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# -------- 记录设置 --------
# 只在主进程设置日志
if Aclrtor.is_main_process:
    # 记录训练日志
    log_dir = os.path.join(args.save_path, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    custom_file_name = f"_{current_time}.tfevents"
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=custom_file_name)
    # 记录配置文件
    config_save_path = save_config(args.cfg, args.save_path, args.save_name)
    logging.info(f"配置文件路径：{config_save_path}")
    logging.info(str(config))  # 打印配置文件参数信息
    logging.info(str(args.t))

    # 损失写入csv
    csv_writer = EpochCSVLogger(
        save_path=args.save_path,
        filename=f'{args.save_name}.csv',
        resume=args.resume,
    )
else:
    writer = None

# ======== 设置模型、优化器、学习率调度器 ========
denoiser = Unet3D.Unet3D(
    init_dim=args.init_dim,
    base_channels=args.base_channels,
    dim_mults=args.dim_mults,
    attn_heads=args.attn_heads,
    flash_attn=args.flash_attn,
    cond_diffusion=args.cond_diffusion,
    full_attn=tuple(args.full_attn),
)
# 获取所有配置中 enabled 为 True 的损失项名称
active_loss_names = [name for name, cfg in args.loss_config.items() if cfg.get('enabled', False)]

# 确保列表顺序固定，避免 CSV 列顺序错乱
active_loss_names.sort()
logging.info(f"Detected active losses for logging: {active_loss_names}")

rf_model = RectifiedFlow_DeEq.RectifiedFlow(
    denoiser,
    input_size=[args.patch_size, args.patch_size, args.slice_count],  # 传入处理的图像尺寸
    num_sample_steps=args.num_sample_steps,  # 采样步长 default
    clip_or_not=True,  # 数值裁剪，防止输出数值爆炸
    clip_range=args.clip_range,  # 裁剪上下限
    sigma_data=args.sigma_data,  # 数据标准差（保留，用于可选归一化）
    noise_scale=args.noise_scale,  # 噪声先验标准差
    init_type=args.init_type,  # 噪声初始化类型
    T=args.T,  # 时间上限（官方=1.0）
    eps=args.eps,  # 时间下限
    loss_config=args.loss_config,  # 损失配置
    use_ode_sampler=args.use_ode_sampler,  # 采样器类型
    ode_tol=args.ode_tol,  # RK45容差
    sigma_variance=args.sigma_variance,  # 扩散转换参数（0=纯ODE）
    device=device,
    deformation_config=args.deformation,  # [DeEq]
)

ema = EMA(rf_model, beta=0.95, update_every=args.ema_update_every)  # EMA蒸馏模型
# 优化器：根据模式决定是否需要优化 RegNet
if args.deformation.get('mode') == 'registration':
    # 联合训练：RegNet + RF
    optimizer = torch.optim.Adam(
        list(rf_model.parameters()),  # 包含 reg_module 参数
        lr=args.lr,
        betas=(0.9, 0.99)
    )
else:
    # 仅训练 RF
    optimizer = torch.optim.Adam(
        rf_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99)
    )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=args.lr_decay)

# ======== 创建训练、测试数据集加载器 ========
train_sampler = np.load(join(args.sampler, f'train_idx_k={args.fold}.npy'))[:]    # todo
test_sampler = np.load(join(args.sampler, f'test_idx_k={args.fold}.npy'))[:]
train_set = NCCT2CTA(
    data_dir=args.data_dir,
    sampler=train_sampler[:],  # 训练集索引
    vol_size=args.vol_size,
    slice_start=args.slice_start,  # 在0-80层之间随机选择起始点，确保覆盖全体积。
    slice_count=args.slice_count,  # 一个patch包含的切片数量=patch的z方向长度
    patch_size=args.patch_size,  # patch平面内的尺寸
    patch_stride=args.patch_stride,  # 平面内patch的滑动步长
    grid_patch_num=args.grid_patch_num,  # 采样策略，从平面的规则网格中取4个（无重叠patch数量=(448//224)**2）
    random_patch_num=args.random_patch_num,  # 全图范围内额外随机取4个patch，不受 patch_stride 限制，可重叠，增加数据多样性
    # 每个病人总patch数 = grid + random
    hu_range=(args.hu_min, args.hu_max),  # CT typical range
    norm_factor=args.norm_factor,
    augment=args.augment,
    augment_prob=args.augment_prob,
)
val_set = NCCT2CTA(
    data_dir=args.data_dir,
    sampler=test_sampler[:],  # 训练集索引
    vol_size=args.vol_size,
    slice_start=args.val_slice_start,  # 在0-80层之间随机选择起始点，确保覆盖全体积。
    slice_count=args.val_slice_count,  # 一个patch包含的切片数量=patch的z方向长度
    patch_size=args.patch_size,  # patch平面内的尺寸
    patch_stride=args.patch_stride,  # 平面内patch的滑动步长
    grid_patch_num=args.grid_patch_num,  # 采样策略，从平面的规则网格中取4个（无重叠patch数量=(448//224)**2）
    random_patch_num=args.random_patch_num,  # 全图范围内额外随机取4个patch，不受 patch_stride 限制，可重叠，增加数据多样性
    # 每个病人总patch数 = grid + random
    hu_range=(args.hu_min, args.hu_max),  # CT typical range
    norm_factor=args.norm_factor,
    augment=False,
)
train_dl = DataLoader(train_set, batch_size=args.bs, num_workers=args.num_workers)
val_dl = DataLoader(val_set, batch_size=args.bs, num_workers=0)

# ---- 使用accelerate准备所有组件 ----
if Aclrtor.is_main_process:
    print(f"ASCEND_VISIBLE_DEVICES: {os.environ.get('ASCEND_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")

rf_model, denoiser, ema, optimizer, train_dl, val_dl, scheduler = Aclrtor.prepare(
    rf_model, denoiser, ema, optimizer, train_dl, val_dl, scheduler)

# ---- 计算总训练步数（用于ETA估计） ----
start_time = time.time()
# 总步数 = 数据长度 × 轮次数
total_steps = len(train_dl) * args.max_epoch
logging.info(f"Total training steps: {total_steps} ({len(train_dl)}×{args.max_epoch} epochs)")

# ---- 断点恢复 ----
start_epoch, global_steps = 1, 1  # 默认从头
if args.resume and Aclrtor.is_main_process:
    unwrap_model = Aclrtor.unwrap_model(rf_model)
    ckpt_dir = join(args.save_path, 'checkpoints')
    start_epoch, global_steps = load_ckpt(device=device, unwrap_model=unwrap_model,
                                          opt=optimizer, sch=scheduler, ckpt_dir=ckpt_dir, accelerator=Aclrtor)
    start_epoch += 1  # 从下一轮开始

# ---- 开始训练循环 ----
val_loss = 10  # 巨大初值
vis_dl = cycle(val_dl)
for epoch in range(start_epoch, args.max_epoch + 1):
    with tqdm(
            train_dl,
            disable=not Aclrtor.is_main_process,
            desc=f"Epoch {epoch}/{args.max_epoch}",
            ncols=120,
    ) as pbar:
        avg_loss, global_steps, avg_loss_dict = train_one_epoch(
            Aclrtor, rf_model, optimizer, pbar,
            global_steps, start_time, total_steps,
            writer=writer  # 传入writer用于实时记录
        )
        ema.update()
        scheduler.step()
        # accelerator.wait_for_everyone()
        Aclrtor.clip_grad_norm_(rf_model.parameters(), args.max_grad_norm)

    train_loss = sum(avg_loss) / len(avg_loss)
    print(f'train loss={train_loss:.4f}, lr={scheduler.get_last_lr()[0]:.1e}')

    # 打印分项损失
    if Aclrtor.is_main_process:
        loss_detail_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_loss_dict.items()])
        print(f'  Loss detail: {loss_detail_str}')

    if divisible_by(epoch, args.val_every):
        val_loss_list, val_loss_dict = test_one_epoch(Aclrtor, rf_model, val_dl)
        val_loss = val_loss_list[0] if isinstance(val_loss_list, list) else val_loss  # 取pixel损失作为总代表

        # 打印验证分项
        if Aclrtor.is_main_process:
            val_detail_str = ", ".join([f"val_{k}={v:.4f}" for k, v in val_loss_dict.items()])
            print(f'  Val detail: {val_detail_str}')
        torch.cuda.empty_cache()

    # 记录总损失和分项
    if Aclrtor.is_main_process:
        writer.add_scalar('Loss_epoch/total', train_loss, epoch)
        for name, value in avg_loss_dict.items():
            writer.add_scalar(f'Loss_epoch/{name}', value, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    # -------- 结果保存 --------
    # 每轮保存一次模型
    save_dir = join(args.save_path, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    if Aclrtor.is_main_process:
        save_epoch_ckpt = divisible_by(epoch, args.save_ckpt_every)
        save_ckpt(save_dir, epoch, global_steps, Aclrtor, rf_model,
                  optimizer, scheduler, ema, save_epoch_ckpt)
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    writer.add_scalar('Train_loss', train_loss, epoch)
    writer.add_scalar('Val_loss', val_loss, epoch)

    # 每 vis_every 轮画图
    if divisible_by(epoch, args.vis_every):
        batch = next(vis_dl)
        torch.cuda.empty_cache()  # 先清理再可视化采样
        cond = batch['cond'].to(device)
        target = batch['target'].to(device)
        visualize_sample(rf_model, ema, args.save_path, args.vis_samples_num, cond, target, epoch)
        # 可视化后清理
        del cond, target
        torch.cuda.empty_cache()

    if Aclrtor.is_main_process:
        # ------ 损失记录 ------
        log_dict = {
            'epoch': epoch,
            'train_total': train_loss,
            'learning_rate': scheduler.get_last_lr()[0],
            'val_total': val_loss,
        }

        # 填充训练集各分项损失
        for name in active_loss_names:
            log_dict[f'train_{name}'] = avg_loss_dict.get(name, 0.0)

        # 填充验证集各分项损失
        if divisible_by(epoch, args.val_every):
            for name in active_loss_names:
                log_dict[f'val_{name}'] = val_loss_dict.get(name, 0.0)
        else:
            for name in active_loss_names:
                log_dict[f'val_{name}'] = 0.0  # 或 np.nan

        # 一行代码完成记录（自动处理覆盖、列排序、写入）
        csv_writer.add_epoch(**log_dict)

        # 每轮结束时重新初始化current_x0_file等变量 at the end of each epoch, call on_epoch_end，重新随机patch的位置
        train_set.on_epoch_end()  # todo
        val_set.on_epoch_end()

# 等待所有进程完成
Aclrtor.wait_for_everyone()

if Aclrtor.is_main_process:
    print(f"训练完成")

