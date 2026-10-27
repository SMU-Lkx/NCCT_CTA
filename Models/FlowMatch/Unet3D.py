# Models/FlowMatch/Unet3D.py
# Modified for MeanFlow project
# [3D-HWD] Modified from 2D to 3D for volumetric data (B, C, H, W, D)
# [3D-HWD] D axis is preserved (no downsampling), only H/W are downsampled via stride=(2,2,1)
"""3D Model architectures with D-axis preserved, dimension order (B,C,H,W,D)"""
import numpy as np
import torch
import math
from torch.nn.functional import silu

from Models.FlowMatch.groupnorm import group_norm


# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer. [Unchanged for 3D]

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# [3D-HWD] Convolutional layer with optional up/downsampling for 3D data.
# Dimension order: (B, C, H, W, D). D axis is preserved, only H/W are downsampled via stride=(2,2,1)

class Conv3d(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=[1, 1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
                 ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel * kernel,
                           fan_out=out_channels * kernel * kernel * kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel, kernel],
                                                     **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(
            weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None

        # [3D-HWD] 2D filter for H/W only: [len, len, 1], D direction length=1 (no actual filtering)
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(2)  # [len, len, 1]
        f = f.unsqueeze(0).unsqueeze(1) / f.sum().square()
        # [3D-HWD] stride=(H,W,D): H/W downsample, D preserved
        self.stride = (2, 2, 1) if (up or down) else (1, 1, 1)

        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[2] - 1) // 2 if f is not None else 0   # todo H/W 方向

        if self.fused_resample and self.up and w is not None:
            # [3D-HWD] conv_transpose3d with stride=(2,2,1), D preserved
            x = torch.nn.functional.conv_transpose3d(x, f.mul(4).tile([self.in_channels, 1, 1, 1, 1]),
                                                     groups=self.in_channels, stride=(2, 2, 1),
                                                     padding=(max(f_pad - w_pad, 0), max(f_pad - w_pad, 0), 0))
            x = torch.nn.functional.conv3d(x, w, padding=(max(w_pad - f_pad, 0), max(w_pad - f_pad, 0), w_pad))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv3d(x, w, padding=(w_pad + f_pad, w_pad + f_pad, w_pad))
            x = torch.nn.functional.conv3d(x, f.tile([self.out_channels, 1, 1, 1, 1]), groups=self.out_channels,
                                           stride=(2, 2, 1))
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose3d(x, f.mul(4).tile([self.in_channels, 1, 1, 1, 1]),
                                                         groups=self.in_channels, stride=(2, 2, 1),
                                                         padding=(f_pad, f_pad, 0))
            if self.down:
                x = torch.nn.functional.conv3d(x, f.tile([self.in_channels, 1, 1, 1, 1]), groups=self.in_channels,
                                               stride=(2, 2, 1), padding=(f_pad, f_pad, 0))
            if w is not None:
                x = torch.nn.functional.conv3d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# [3D-HWD] Group normalization for 3D. [Unchanged]

class GroupNorm3D(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype),
                       eps=self.eps)
        return x


# ----------------------------------------------------------------------------
# [3D-HWD] QKV Attention for 3D. [Unchanged]

class QKVAttention(torch.nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)


# ----------------------------------------------------------------------------
# [3D-HWD] Unified U-Net block for 3D with optional up/downsampling.
# Dimension order: (B, C, H, W, D). D axis is preserved in up/down operations.

class UNetBlock3D(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1, 1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = float(skip_scale)
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm3D(num_channels=in_channels, eps=eps)
        self.conv0 = Conv3d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down,
                            resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm3D(num_channels=out_channels, eps=eps)
        self.conv1 = Conv3d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv3d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down,
                               resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm3D(num_channels=out_channels, eps=eps)
            self.qkv = Conv3d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1,
                              **(init_attn if init_attn is not None else init))
            self.proj = Conv3d(in_channels=out_channels, out_channels=out_channels, kernel=1,
                               **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        # [3D-HWD] emb shape (B, emb_dim) -> unsqueeze to (B, emb_dim, 1, 1, 1) for (B,C,H,W,D)
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            qkv = self.qkv(self.norm2(x))
            qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], -1)
            a = QKVAttention(n_heads=self.num_heads)(qkv)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding. [Unchanged]

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding. [Unchanged]

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# [3D-HWD] SongUNet for 3D volumetric data with D-axis preserved.
# Dimension order: (B, C, H, W, D). D axis is kept constant, only H/W are downsampled.

class SongUNet3D(torch.nn.Module):
    def __init__(self,
                 input_size,  # [3D-HWD] tuple (H, W, D) or int for H=W, D from data
                 in_channels,
                 out_channels,
                 label_dim=0,
                 augment_dim=0,

                 model_channels=128,
                 channel_mult=[1, 2, 2, 2],
                 channel_mult_emb=4,
                 num_blocks=4,
                 attn_resolutions=[16],  # [3D-HWD] H/W resolutions for attention
                 dropout=0.10,
                 label_dropout=0,

                 embedding_type='positional',
                 channel_mult_noise=1,
                 encoder_type='standard',
                 decoder_type='standard',
                 resample_filter=[1, 1],
                 ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout

        # [3D-HWD] input_size: if int, H=W=input_size, D from data
        # if tuple (H, W), D from data
        # if tuple (H, W, D), fixed D
        if isinstance(input_size, int):
            self.input_h = self.input_w = input_size
            self.input_d = None  # D is dynamic, from data
        elif len(input_size) == 2:
            self.input_h, self.input_w = input_size
            self.input_d = None
        else:
            self.input_h, self.input_w, self.input_d = input_size

        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping. [Unchanged]
        self.map_noise = PositionalEmbedding(num_channels=noise_channels,
                                             endpoint=True) if embedding_type == 'positional' else FourierEmbedding(
            num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels * 2, bias=False,
                                  **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels * 2, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # [3D-HWD] Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            # [3D-HWD] Only H/W change, D stays constant
            res_h = self.input_h >> level
            res_w = self.input_w >> level
            res_d = self.input_d if self.input_d is not None else 'D'  # D is preserved
            res_key = f'{res_h}x{res_w}x{res_d}'  # e.g., '32x32xD' or '32x32x70'

            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res_key}_conv'] = Conv3d(in_channels=cin, out_channels=cout, kernel=3,
                                                     **init)
            else:
                self.enc[f'{res_key}_down'] = UNetBlock3D(in_channels=cout, out_channels=cout, down=True,
                                                          **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res_key}_aux_down'] = Conv3d(in_channels=caux, out_channels=caux, kernel=0, down=True,
                                                             resample_filter=resample_filter)
                    self.enc[f'{res_key}_aux_skip'] = Conv3d(in_channels=caux, out_channels=cout, kernel=1,
                                                             **init)
                if encoder_type == 'residual':
                    self.enc[f'{res_key}_aux_residual'] = Conv3d(in_channels=caux, out_channels=cout, kernel=3,
                                                                 down=True, resample_filter=resample_filter,
                                                                 fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                # [3D-HWD] Attention only on H/W resolution
                attn = (res_h in attn_resolutions)
                self.enc[f'{res_key}_block{idx}'] = UNetBlock3D(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # [3D-HWD] Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res_h = self.input_h >> level
            res_w = self.input_w >> level
            res_d = self.input_d if self.input_d is not None else 'D'
            res_key = f'{res_h}x{res_w}x{res_d}'

            if level == len(channel_mult) - 1:
                self.dec[f'{res_key}_in0'] = UNetBlock3D(in_channels=cout, out_channels=cout, attention=True,
                                                         **block_kwargs)
                self.dec[f'{res_key}_in1'] = UNetBlock3D(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res_key}_up'] = UNetBlock3D(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res_h in attn_resolutions)
                self.dec[f'{res_key}_block{idx}'] = UNetBlock3D(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res_key}_aux_up'] = Conv3d(in_channels=out_channels, out_channels=out_channels,
                                                           kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res_key}_aux_norm'] = GroupNorm3D(num_channels=cout, eps=1e-6)
                self.dec[f'{res_key}_aux_conv'] = Conv3d(in_channels=cout, out_channels=out_channels, kernel=3,
                                                         **init_zero)

    def forward(self, x, time_steps, aug_cond=None):
        augment_labels = aug_cond

        assert type(time_steps) is tuple and len(time_steps) == 2, "time_steps must be a tuple of (t, h)"
        t, h = time_steps

        # Mapping. [Unchanged]
        emb_t = self.map_noise(t)
        emb_t = emb_t.reshape(emb_t.shape[0], 2, -1).flip(1).reshape(*emb_t.shape)
        emb_h = self.map_noise(h)
        emb_h = emb_h.reshape(emb_h.shape[0], 2, -1).flip(1).reshape(*emb_h.shape)
        emb = torch.cat([emb_t, emb_h], dim=1)

        if self.map_label is not None:
            raise NotImplementedError
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # [3D-HWD] Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock3D) else block(x)
                skips.append(x)

        # [3D-HWD] Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux
