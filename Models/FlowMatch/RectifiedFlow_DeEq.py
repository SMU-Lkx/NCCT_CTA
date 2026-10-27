# Models/FlowMatch/RectifiedFlow_DeEq.py
"""
Rectified Flow for 3D CT Motion Artifact Correction.
封装整形流模型的核心数学、训练step和采样逻辑。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from Models.FlowMatch.registration import RegistrationModule, gradient_norm, NCC


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# [DeEq] 刚性变形等变性支持
class RandomAffineDeformation3D:
    def __init__(
            self,
            max_translation: float = 0.05,
            max_rotation: float = 5.0,
            prob: float = 1.0,
    ):
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.prob = prob

    def __call__(self, shape, device):
        B, C, D, H, W = shape
        if torch.rand(1).item() > self.prob:
            return None

        matrices = []
        for _ in range(B):
            # 平移（归一化到[-1,1]坐标系）
            tx = (torch.rand(1).item() * 2 - 1) * self.max_translation
            ty = (torch.rand(1).item() * 2 - 1) * self.max_translation
            tz = (torch.rand(1).item() * 2 - 1) * self.max_translation

            # 旋转（度转弧度）
            rx = (torch.rand(1).item() * 2 - 1) * self.max_rotation * math.pi / 180
            ry = (torch.rand(1).item() * 2 - 1) * self.max_rotation * math.pi / 180
            rz = (torch.rand(1).item() * 2 - 1) * self.max_rotation * math.pi / 180

            # 旋转矩阵（ZYX顺序）
            Rx = torch.tensor([
                [1, 0, 0],
                [0, math.cos(rx), -math.sin(rx)],
                [0, math.sin(rx), math.cos(rx)]
            ], dtype=torch.float32)
            Ry = torch.tensor([
                [math.cos(ry), 0, math.sin(ry)],
                [0, 1, 0],
                [-math.sin(ry), 0, math.cos(ry)]
            ], dtype=torch.float32)
            Rz = torch.tensor([
                [math.cos(rz), -math.sin(rz), 0],
                [math.sin(rz), math.cos(rz), 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            R = Rz @ Ry @ Rx

            # 组合仿射矩阵 [R | t]，grid_sample期望(N,3,4)
            t_vec = torch.tensor([tx, ty, tz], dtype=torch.float32)
            affine = torch.cat([R, t_vec.unsqueeze(1)], dim=1)  # (3, 4)
            matrices.append(affine)

        affine_matrices = torch.stack(matrices).to(device)  # (B, 3, 4)

        # 生成采样网格
        grid = F.affine_grid(
            affine_matrices,
            size=shape,
            align_corners=False
        )  # (B, D, H, W, 3)

        return grid

    @staticmethod
    def apply(image, grid):
        if grid is None:
            return image
        return F.grid_sample(
            image, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )


class RandomElasticDeformation3D:
    def __init__(
            self,
            max_displacement_pixels: float = 5.0,
            elastic_std: float = 0.1,
            prob: float = 1.0,
    ):
        self.max_displacement_pixels = max_displacement_pixels
        self.elastic_std = elastic_std
        self.prob = prob

    def __call__(self, shape, device):
        B, C, D, H, W = shape

        # 计算归一化速度场幅度
        # 目标：像素位移 = v_norm * (size/2) = max_displacement_pixels
        # 所以 v_norm = max_displacement_pixels / (size/2)
        max_v_z = self.max_displacement_pixels / (D / 2)
        max_v_y = self.max_displacement_pixels / (H / 2)
        max_v_x = self.max_displacement_pixels / (W / 2)

        # 采样速度场，然后裁剪到 [-max_v, max_v]
        v = torch.randn(B, 3, D, H, W, device=device)
        v = torch.tanh(v)  # 映射到 [-1, 1]
        v[:, 0] *= max_v_z
        v[:, 1] *= max_v_y
        v[:, 2] *= max_v_x

        # 2. 高斯平滑（模拟解剖合理性：邻近像素位移相似）
        # 使用多次平均池化近似高斯平滑
        kernel_size = max(3, int(self.elastic_std * min(D, H, W)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        for _ in range(3):  # 多次平滑
            v = F.avg_pool3d(
                v,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                count_include_pad=False
            )

        # 3. 从速度场积分得到位移场（小变形近似：直接作为位移）
        # 完整实现需 Scaling-and-Squaring，这里简化（小变形时exp(v)≈I+v）
        # 归一化到 [-1, 1] 坐标系
        displacement = v.clone()
        displacement[:, 0] = displacement[:, 0] / (D - 1) * 2  # z
        displacement[:, 1] = displacement[:, 1] / (H - 1) * 2  # y
        displacement[:, 2] = displacement[:, 2] / (W - 1) * 2  # x

        # 4. 生成采样网格
        # 创建标准网格 [-1, 1]
        grid_z = torch.linspace(-1, 1, D, device=device)
        grid_y = torch.linspace(-1, 1, H, device=device)
        grid_x = torch.linspace(-1, 1, W, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D, H, W, 3)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B, D, H, W, 3)

        # 加上位移（注意：grid_sample 的 grid 最后一维是 (x, y, z)）
        # displacement 是 (B, 3, D, H, W)，需 permute
        disp_perm = displacement.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
        grid = base_grid + disp_perm

        return grid

    @staticmethod
    def apply(image, grid):
        if grid is None:
            return image
        return F.grid_sample(
            image, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )


class NoiseSampler:
    def __init__(self, noise_scale: float = 1.0, init_type: str = 'gaussian'):
        self.noise_scale = noise_scale
        self.init_type = init_type

    def __call__(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        if self.init_type == 'gaussian':
            return torch.randn(shape, device=device) * self.noise_scale
        elif self.init_type == 'uniform':
            return (torch.rand(shape, device=device) * 2 - 1) * self.noise_scale
        elif self.init_type == 'sphere':
            z = torch.randn(shape, device=device)
            z = z / (z.norm(dim=tuple(range(1, z.ndim)), keepdim=True) + 1e-12)
            return z * self.noise_scale
        else:
            raise ValueError(f"Unknown init_type: {self.init_type}")


class TimeScheduler:
    def __init__(self, T: float = 1.0, eps: float = 1e-3):
        self.T = T
        self.eps = eps

    def sample_train(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device) * (self.T - self.eps) + self.eps

    def sample_step(self, step: int, total_steps: int, device: torch.device) -> torch.Tensor:
        num_t = step / total_steps * (self.T - self.eps) + self.eps
        return torch.ones(1, device=device) * num_t

    def normalize_for_model(self, t: torch.Tensor) -> torch.Tensor:
        return t * 999.0


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        model: nn.Module,                      # 去噪器/速度预测器 (Unet3D)
        input_size: Tuple[int, int, int],      # (D, H, W) 或 (Z, H, W)
        num_sample_steps: int = 50,            # 采样步数 N，对应官方 sample_N
        clip_or_not: bool = True,              # 是否裁剪输出
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        sigma_data: float = 0.5,               # 数据标准差
        noise_scale: float = 1.0,              # 噪声先验标准差，对应官方 init_noise_scale
        init_type: str = 'gaussian',           # 噪声初始化类型，对应官方 init_type
        T: float = 1.0,                        # 最大时间步 = 1.0
        eps: float = 1e-3,                     # 最小时间步（数值稳定性）
        loss_config: Optional[Dict[str, Any]] = None,  # 损失配置（兼容EDM接口）
        use_ode_sampler: str = 'euler',        # 'euler' | 'rk45'，对应官方 use_ode_sampler
        ode_tol: float = 1e-5,                 # RK45容差，对应官方 ode_tol
        sigma_variance: float = 0.0,           # 扩散转换参数，对应官方 sigma_var
        device: str = 'cuda',
        deformation_config: Optional[Dict[str, Any]] = None,    # [DeEq]
    ):
        super().__init__()
        self.model = model
        self.input_size = list(input_size)
        self.num_sample_steps = num_sample_steps
        self.clip_or_not = clip_or_not
        self.clip_range = clip_range
        self.sigma_data = sigma_data
        self.T = T
        self.eps = eps
        self.use_ode_sampler = use_ode_sampler
        self.ode_tol = ode_tol
        self.sigma_variance = sigma_variance
        self.device = device

        self.noise_sampler = NoiseSampler(noise_scale=noise_scale, init_type=init_type)
        self.time_scheduler = TimeScheduler(T=T, eps=eps)

        self.loss_config = default(loss_config, {
            'pixel': {
                'enabled': True,
                'weight': 1.0,
                'type': 'MSE'
            },
            'alignment': {
                'enabled': False,
                'weight': 5.0,
            },
            'registration_ncc': {
                'enabled': False,
                'weight': 10.0,
            },
            'registration_smooth': {
                'enabled': False,
                'weight': 10.0,
            },
            # # [DeEq] 默认等变性配置
            # 'equivariance': {
            #     'enabled': False,
            #     'weight': 0.1,
            #     'max_translation': 0.05,
            #     'max_rotation': 5.0,
            #     'prob': 1.0,
            # }
        })

        self.deformation_config = default(deformation_config, {
            'mode': 'none',
        })
        self.deformation_mode = self.deformation_config.get('mode', 'none')

        if self.deformation_mode == 'rigid':
            cfg = self.deformation_config.get('rigid', {})
            self.deformation = RandomAffineDeformation3D(
                max_translation=cfg.get('max_translation', 0.05),
                max_rotation=cfg.get('max_rotation', 5.0),
                prob=cfg.get('prob', 1.0),
            )
            self.eq_weight = cfg.get('weight', 0.1)

        elif self.deformation_mode == 'elastic':
            cfg = self.deformation_config.get('elastic', {})
            self.deformation = RandomElasticDeformation3D(
                max_displacement_pixels=cfg.get('max_displacement_pixels', 5.0),
                elastic_std=cfg.get('elastic_std', 0.1),
                prob=cfg.get('prob', 1.0),
            )
            self.eq_weight = cfg.get('weight', 0.1)


        elif self.deformation_mode == 'registration':
            cfg = self.deformation_config.get('registration', {})
            max_disp = cfg.get('max_disp', 0.5)
            self.reg_module = RegistrationModule(
                win=cfg.get('ncc_window', 9),
                n_steps=cfg.get('ssv_steps', 7),
                max_disp=max_disp,

            )
            self.align_weight = self.loss_config.get('alignment', {}).get('weight', 5.0)
            self.reg_weight = self.loss_config.get('registration_ncc', {}).get('weight', 10.0)
            self.smooth_weight = self.loss_config.get('registration_smooth', {}).get('weight', 10.0)
            # 一次性启用，确保 active_loss_names 能正确捕获
            if self.align_weight > 0:
                self.loss_config['alignment']['enabled'] = True
            if self.reg_weight > 0:
                self.loss_config['registration_ncc']['enabled'] = True
            if self.smooth_weight > 0:
                self.loss_config['registration_smooth']['enabled'] = True
        else:
            self.deformation = None
            self.reg_module = None
        # # [DeEq] 从 loss_config 读取等变性配置
        # eq_cfg = self.loss_config.get('equivariance', {})
        # if eq_cfg.get('enabled', False):
        #     mode = eq_cfg.get('mode', 'rigid')  # [DeEq] 新增 mode 参数
        #
        #     if mode == 'rigid':
        #         self.deformation = RandomAffineDeformation3D(
        #             max_translation=eq_cfg.get('max_translation', 0.05),
        #             max_rotation=eq_cfg.get('max_rotation', 5.0),
        #             prob=eq_cfg.get('prob', 1.0),
        #         )
        #     elif mode == 'elastic':
        #         self.deformation = RandomElasticDeformation3D(
        #             max_displacement_pixels=eq_cfg.get('max_displacement_pixels', 20.0),
        #             elastic_std=eq_cfg.get('elastic_std', 0.1),
        #             prob=eq_cfg.get('prob', 1.0),
        #         )
        #     else:
        #         raise ValueError(f"[DeEq] Unknown mode: {mode}")

    def _get_z0(self, target: torch.Tensor) -> torch.Tensor:
        return self.noise_sampler(target.shape, target.device)

    def _interpolate(self, z0: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_expand = t.view(-1, 1, 1, 1, 1)
        z_t = (1.0 - t_expand) * z0 + t_expand * target
        return z_t

    def _velocity_target(self, z0: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return target - z0

    def _model_forward(self, x: torch.Tensor, t: torch.Tensor,
                       condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_norm = self.time_scheduler.normalize_for_model(t)
        return self.model(x, t_norm, condition=condition)

    def _sigma_t(self, t: float) -> float:
        if self.sigma_variance <= 0.0:
            return 0.0
        return (1.0 - t) * self.sigma_variance

    def _apply_deformation(self, image, grid_or_displacement):
        if self.deformation_mode in ['rigid', 'elastic']:
            if self.deformation_mode == 'rigid':
                return RandomAffineDeformation3D.apply(image, grid_or_displacement)
            else:
                return RandomElasticDeformation3D.apply(image, grid_or_displacement)
        elif self.deformation_mode == 'registration':
            return self.reg_module.warp(image, grid_or_displacement)
        else:
            return image

    def forward(
            self,
            target: torch.Tensor,
            condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        batch_size = target.shape[0]
        device = target.device

        # 1. 标准 RF 前向
        z0 = self._get_z0(target)
        t = self.time_scheduler.sample_train(batch_size, device)
        z_t = self._interpolate(z0, target, t)
        v_target = self._velocity_target(z0, target)
        v_pred = self._model_forward(z_t, t, condition=condition)

        # 2. Pixel loss
        pixel_weight = self.loss_config.get('pixel', {}).get('weight', 1.0)
        pixel_loss = F.mse_loss(v_pred, v_target)
        loss = pixel_weight * pixel_loss
        loss_dict = {'pixel': pixel_loss.item()}

        # 3. 变形等变损失
        if self.training and self.deformation_mode != 'none':

            if self.deformation_mode in ['rigid', 'elastic']:
                grid = self.deformation(z_t.shape, device)
                if grid is not None:
                    # 路径1：先变形 z_t，再预测速度
                    z_t_deformed = self._apply_deformation(z_t, grid)
                    cond_deformed = self._apply_deformation(condition, grid) if condition is not None else None
                    v_pred_deformed = self._model_forward(z_t_deformed, t, condition=cond_deformed)

                    # 路径2：先预测速度，再变形输出
                    v_pred_warped = self._apply_deformation(v_pred, grid)

                    # 等变损失
                    eq_loss = F.mse_loss(v_pred_deformed, v_pred_warped)
                    loss = loss + self.eq_weight * eq_loss
                    loss_dict['equivariance'] = eq_loss.item()

            elif self.deformation_mode == 'registration':  # [修正] 改为 elif，移出上面的 if

                # ====== 配准网络引导等变 ======
                # 3.1 预测变形场
                displacement, condition_aligned, svf = self.reg_module(condition, target)

                # [修正] 打印更详细的统计量...
                print(f"  disp_mean={displacement.abs().mean():.4f}, "
                      f"disp_max={displacement.abs().max():.4f}, "
                      f"disp_std={displacement.std():.4f}, "
                      f"max_disp_param={F.softplus(self.reg_module.regnet.max_disp).item():.4f}")

                # [修正] 关键改动：改为图像级对齐损失（Alignment Loss）
                # 两条路径的输出都直接监督到target

                t_expand = t.view(-1, 1, 1, 1, 1)

                # 路径1：先配准条件，再预测速度（Synthesis After Registration）
                # z_t 用配准后的条件
                z_t_warped = self.reg_module.warp(z_t, displacement)
                v_pred_after = self._model_forward(z_t_warped, t, condition=condition_aligned)
                # 从z_t_warped出发，用v_pred_after走一步，近似预测图像
                x_pred_after = z_t_warped + (1.0 - t_expand) * v_pred_after

                # 路径2：先预测速度，再配准输出（Synthesis Before Registration）
                # 从z_t出发，用原始v_pred走一步，再配准
                x_pred_before_raw = z_t + (1.0 - t_expand) * v_pred
                x_pred_before = self.reg_module.warp(x_pred_before_raw, displacement)

                # [修正] 图像级Alignment Loss：两条路径都应该接近target图像
                align_loss = F.mse_loss(x_pred_after, target) + F.mse_loss(x_pred_before, target)
                loss = loss + self.align_weight * align_loss
                loss_dict['alignment'] = align_loss.item()

                # 3.5 配准损失：NCC(condition_aligned, target)
                reg_loss = self.reg_module.compute_ncc_loss(condition_aligned, target)
                loss = loss + self.reg_weight * reg_loss
                loss_dict['registration_ncc'] = reg_loss.item()

                # 3.6 变形场平滑正则
                smooth_loss = gradient_norm(displacement)
                loss = loss + self.smooth_weight * smooth_loss
                loss_dict['registration_smooth'] = smooth_loss.item()
        # # [DeEq] 从 loss_config 读取配置
        # eq_cfg = self.loss_config.get('equivariance', {})
        # if self.training and eq_cfg.get('enabled', False):
        #     grid = self.deformation(z_t.shape, device)
        #
        #     if grid is not None:
        #         # 路径1：先变形 z_t，再预测速度
        #         z_t_deformed = RandomAffineDeformation3D.apply(z_t, grid)
        #         cond_deformed = (
        #             RandomAffineDeformation3D.apply(condition, grid)
        #             if condition is not None else None
        #         )
        #         v_pred_deformed = self._model_forward(
        #             z_t_deformed, t, condition=cond_deformed
        #         )
        #
        #         # 路径2：先预测速度，再变形输出
        #         v_pred_warped = RandomAffineDeformation3D.apply(v_pred, grid)
        #
        #         # 等变性损失
        #         eq_loss = F.mse_loss(v_pred_deformed, v_pred_warped)
        #
        #         # [DeEq] 从 loss_config 读取权重
        #         eq_weight = eq_cfg.get('weight', 0.1)
        #         loss = loss + eq_weight * eq_loss
        #         loss_dict['equivariance'] = eq_loss.item()
        #         loss_dict['equivariance_weighted'] = eq_weight * eq_loss.item()

        loss_dict['total'] = loss.item()
        return loss, loss_dict

    @torch.no_grad()
    def sample(
            self,
            condition: Optional[torch.Tensor] = None,
            batch_size: int = 1,
            return_nfe: bool = True,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        if exists(condition):
            batch_size = condition.shape[0]
            device = condition.device
        else:
            device = self.device

        # 初始噪声 z_0
        shape = [batch_size, 1] + self.input_size  # (B, 1, D, H, W)
        x = self.noise_sampler(shape, device)

        if self.use_ode_sampler == 'euler':
            samples, nfe = self._euler_sample(x, condition)
        elif self.use_ode_sampler == 'rk45':
            samples, nfe = self._rk45_sample(x, condition)
        else:
            raise ValueError(f"Unknown sampler: {self.use_ode_sampler}")

        if self.clip_or_not:
            samples = torch.clamp(samples, *self.clip_range)

        if return_nfe:
            return samples, nfe
        return samples, None

    def _euler_sample(
            self,
            x: torch.Tensor,
            condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        device = x.device
        N = self.num_sample_steps
        dt = 1.0 / N

        for i in range(N):
            num_t = i / N * (self.T - self.eps) + self.eps
            t = torch.ones(x.shape[0], device=device) * num_t

            pred = self._model_forward(x, t, condition=condition)

            if self.sigma_variance > 0.0:
                sigma_t = self._sigma_t(num_t)
                pred = pred + (sigma_t ** 2) / (2 * (self.noise_sampler.noise_scale ** 2) * ((1.0 - num_t) ** 2)) * \
                       (0.5 * num_t * (1.0 - num_t) * pred - 0.5 * (2.0 - num_t) * x)
                x = x + pred * dt + sigma_t * math.sqrt(dt) * torch.randn_like(pred)
            else:
                # 纯ODE：x_{t+dt} = x_t + v * dt
                x = x + pred * dt

        nfe = N
        return x, nfe

    def _rk45_sample(
            self,
            x: torch.Tensor,
            condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        from scipy import integrate
        from util_RectifiedFlow import from_flattened_numpy, to_flattened_numpy

        device = x.device
        shape = x.shape
        rtol = atol = self.ode_tol
        method = 'RK45'

        def ode_func(t, x_flat):
            x_tensor = from_flattened_numpy(x_flat, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=device) * t
            with torch.no_grad():
                drift = self._model_forward(x_tensor, vec_t, condition=condition)
            return to_flattened_numpy(drift)

        solution = integrate.solve_ivp(
            ode_func,
            (self.eps, self.T),
            to_flattened_numpy(x),
            rtol=rtol,
            atol=atol,
            method=method,
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

        return x, nfe

    @torch.no_grad()
    def reflow_sample_pair(
            self,
            condition: Optional[torch.Tensor] = None,
            batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = [batch_size, 1] + self.input_size
        z0 = self.noise_sampler(shape, self.device)
        x1, _ = self._euler_sample(z0, condition)
        if self.clip_or_not:
            x1 = torch.clamp(x1, *self.clip_range)
        return z0, x1