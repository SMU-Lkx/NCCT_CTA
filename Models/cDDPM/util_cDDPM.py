# Models/cDDPM/util_cDDPM.py
import torch
import os
from os.path import join
import logging
import glob
import shutil
from typing import BinaryIO, List, Optional, Union
import pathlib
from torchvision.utils import make_grid
from PIL import Image

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def c_noise(sigma):
    return log(sigma) * 0.25


def load_ckpt(device, unwrap_model, opt, sch, ckpt_dir, ema=None, accelerator=None):
    start_epoch, global_steps = 1, 1
    # 检查目录是否存在
    if not os.path.exists(ckpt_dir):
        logging.info(f"\nCheckpoint directory '{ckpt_dir}' does not exist yet.")
        return start_epoch, global_steps

    # 查找最新的检查点文件
    ckpt_files = glob.glob(join(ckpt_dir, f'lastest_model.pt'))
    if not ckpt_files:
        logging.info(f"No checkpoint files found in '{ckpt_dir}'. Starting from scratch.")
        return start_epoch, global_steps

    # 加载最新的检查点文件
    latest_ckpt = max(ckpt_files, key=os.path.getctime)
    ckpt = torch.load(latest_ckpt, map_location=device)

    unwrap_model.load_state_dict(ckpt['model'])
    start_epoch = ckpt['epoch']
    global_steps = ckpt['global_steps']
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])

    # 【关键】确保所有优化器参数都在正确设备上
    for param_group in opt.param_groups:
        for param in param_group['params']:
            # 确保参数在正确设备上
            if param.device != device:
                # print(f"移动参数从 {param.device} 到 {device}")
                param.data = param.data.to(device)
            # 确保梯度在正确设备上（如果有的话）
            if param.grad is not None and param.grad.device != device:
                param.grad = param.grad.to(device)

    # 移动优化器状态到正确设备
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and value.device != device:
                # print(f"移动优化器状态 {key} 从 {value.device} 到 {device}")
                state[key] = value.to(device)

    if ema is not None:
        ema.load_state_dict(ckpt["ema"])

    if exists(accelerator.scaler) and exists(ckpt['scaler']):
        accelerator.scaler.load_state_dict(ckpt['scaler'])

    return start_epoch, global_steps

def save_ckpt(save_dir, epoch, global_steps, Aclrtor, model, opt, sch, ema=None, save_epoch_ckpt=False):
    data = {
        'epoch': epoch,
        'global_steps': global_steps,
        'model': Aclrtor.get_state_dict(model),
        'optimizer': opt.state_dict(),
        'scheduler': sch.state_dict(),
        'ema': ema.state_dict() if ema is not None else None,
        'scaler': Aclrtor.scaler.state_dict() if exists(Aclrtor.scaler) else None,
        }

    # 每轮保存一次最新模型
    # 先写到临时文件，再原子覆盖，防止并发写坏
    tmp_path = join(save_dir, 'lastest_model.tmp')
    torch.save(data, tmp_path)
    shutil.move(tmp_path, join(save_dir, 'lastest_model.pt'))

    # 只在指定间隔保存epoch模型
    if save_epoch_ckpt:
        torch.save(data, join(save_dir, f'model_epoch{epoch:04d}.pt'))

@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)





