"""
3D Swin Transformer 模型
========================
这是项目的核心 - 3D Swin Transformer实现！

作为新手，你需要理解的关键概念：
1. Patch Embedding: 将输入图像分割成小块（patches）
2. Window Attention: 只在局部窗口内计算注意力（节省计算）
3. Shifted Window: 窗口移位，让不同窗口的信息能交互
4. Hierarchical Structure: 层次化结构，多尺度特征

参考资料：
- Swin Transformer论文: https://arxiv.org/abs/2103.14030
- Video Swin Transformer: https://arxiv.org/abs/2106.13230
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
from einops import rearrange, repeat


# ============================================
# 工具函数
# ============================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample.
    随机深度正则化 - 训练时随机丢弃一些残差连接
    
    参数:
        x: 输入张量
        drop_prob: 丢弃概率
        training: 是否训练模式
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    包装成nn.Module以便在nn.Sequential中使用
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    多层感知机 (MLP)
    Transformer中的前馈网络
    
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout
    
    参数:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度（通常是in_features的4倍）
        out_features: 输出特征维度
        act_layer: 激活函数层
        drop: Dropout概率
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_3d(x, window_size):
    """
    3D窗口分区函数
    将输入张量分割成不重叠的3D窗口
    
    这是3D Swin Transformer的核心操作！
    
    参数:
        x: (B, T, H, W, C) - 输入张量
        window_size: (T_w, H_w, W_w) - 窗口大小
    
    返回:
        windows: (num_windows*B, T_w*H_w*W_w, C) - 窗口化的张量
    """
    B, T, H, W, C = x.shape
    T_w, H_w, W_w = window_size
    
    # 确保输入尺寸能被窗口大小整除
    assert T % T_w == 0 and H % H_w == 0 and W % W_w == 0, \
        f"输入尺寸({T},{H},{W})必须能被窗口大小{window_size}整除"
    
    # 重塑张量以便提取窗口
    # (B, T, H, W, C) -> (B, T//T_w, T_w, H//H_w, H_w, W//W_w, W_w, C)
    x = x.view(B, T // T_w, T_w, H // H_w, H_w, W // W_w, W_w, C)
    
    # 重排维度以将窗口维度放在一起
    # -> (B, T//T_w, H//H_w, W//W_w, T_w, H_w, W_w, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    
    # 展平窗口内的所有位置
    # -> (B * num_windows, T_w*H_w*W_w, C)
    windows = x.view(-1, T_w * H_w * W_w, C)
    
    return windows


def window_reverse_3d(windows, window_size, T, H, W):
    """
    3D窗口逆操作 - 将窗口化的张量恢复为原始形状
    
    这是window_partition_3d的逆操作
    
    参数:
        windows: (num_windows*B, T_w*H_w*W_w, C) - 窗口化张量
        window_size: (T_w, H_w, W_w) - 窗口大小
        T, H, W: 原始时空尺寸
    
    返回:
        x: (B, T, H, W, C) - 恢复后的张量
    """
    T_w, H_w, W_w = window_size
    B = int(windows.shape[0] / (T * H * W / (T_w * H_w * W_w)))
    C = windows.shape[-1]
    
    # 恢复窗口维度
    # (num_windows*B, T_w*H_w*W_w, C) -> (B, T//T_w, H//H_w, W//W_w, T_w, H_w, W_w, C)
    x = windows.view(B, T // T_w, H // H_w, W // W_w, T_w, H_w, W_w, C)
    
    # 重排维度
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    
    # 恢复原始形状
    x = x.view(B, T, H, W, C)
    
    return x


def get_effective_shift_size_3d(x_size, window_size, shift_size):
    """
    根据当前输入尺寸动态调整shift size。

    当某个维度小于等于窗口大小时，该维度只会形成一个窗口，
    此时不应再执行shift，否则只会引入无意义的循环移位。
    """
    effective_shift = []
    for size, window, shift in zip(x_size, window_size, shift_size):
        effective_shift.append(0 if size <= window else shift)
    return tuple(effective_shift)


def compute_attention_mask_3d(x_size, window_size, shift_size, device):
    """
    为Shifted Window Attention构建3D注意力掩膜。
    """
    if not any(shift_size):
        return None

    Tp = ((x_size[0] + window_size[0] - 1) // window_size[0]) * window_size[0]
    Hp = ((x_size[1] + window_size[1] - 1) // window_size[1]) * window_size[1]
    Wp = ((x_size[2] + window_size[2] - 1) // window_size[2]) * window_size[2]

    img_mask = torch.zeros((1, Tp, Hp, Wp, 1), device=device)

    t_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ) if shift_size[0] > 0 else (slice(0, Tp),)
    h_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    ) if shift_size[1] > 0 else (slice(0, Hp),)
    w_slices = (
        slice(0, -window_size[2]),
        slice(-window_size[2], -shift_size[2]),
        slice(-shift_size[2], None),
    ) if shift_size[2] > 0 else (slice(0, Wp),)

    cnt = 0
    for t in t_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, t, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition_3d(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


# ============================================
# 3D窗口注意力机制
# ============================================

class WindowAttention3D(nn.Module):
    """
    3D基于窗口的多头自注意力 (W-MSA)
    
    这是Swin Transformer的核心创新：
    只在局部窗口内计算注意力，大幅降低计算复杂度
    
    支持两种模式：
    1. W-MSA (Window Multi-head Self-Attention): 普通窗口注意力
    2. SW-MSA (Shifted Window Multi-head Self-Attention): 移位窗口注意力
    
    参数:
        dim: 输入特征维度
        window_size: 3D窗口大小 (T, H, W)
        num_heads: 注意力头数
        qkv_bias: 是否在QKV投影中使用偏置
        attn_drop: 注意力权重dropout率
        proj_drop: 输出投影dropout率
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (T, H, W)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 注意力缩放因子
        
        # 相对位置偏置表
        # 这是Swin Transformer的关键创新，学习相对位置信息
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        
        # 初始化相对位置偏置
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # 获取相对位置索引（只需计算一次）
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        
        # 创建3D坐标网格
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing='ij'))  # (3, T, H, W)
        coords_flatten = torch.flatten(coords, 1)  # (3, T*H*W)
        
        # 计算相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (3, T*H*W, T*H*W)
        
        # 映射到非负索引
        relative_coords[0] += self.window_size[0] - 1  # time
        relative_coords[1] += self.window_size[1] - 1  # height
        relative_coords[2] += self.window_size[2] - 1  # width
        
        # 将3D相对坐标映射到1D索引
        relative_coords = (
            relative_coords[0] * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1) +
            relative_coords[1] * (2 * self.window_size[2] - 1) +
            relative_coords[2]
        )  # (T*H*W, T*H*W)
        
        # 注册为buffer（不参与梯度更新，但会保存）
        self.register_buffer("relative_position_index", relative_coords)
        
        # QKV线性投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初始化
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (B*nW, N, C)
               B=batch size, nW=窗口数, N=每窗口token数, C=特征维度
            mask: 注意力掩膜（用于移位窗口），形状 (nW, N, N)
        
        返回:
            output: 输出张量，形状 (B*nW, N, C)
        """
        BnW, N, C = x.shape  # BnW = B * num_windows
        
        # QKV投影: (B*nW, N, C) -> (B*nW, N, 3*C) -> (B*nW, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(BnW, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # q, k, v: 每个都是 (B*nW, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 缩放注意力: (B*nW, num_heads, N, N)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # 矩阵乘法
        
        # 添加相对位置偏置
        # 从lookup table中获取
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        )  # (N, N, num_heads)
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)  # (B*nW, num_heads, N, N)
        
        # 应用掩膜（用于移位窗口）
        if mask is not None:
            # mask: (nW, N, N)
            nW = mask.shape[0]
            attn = attn.view(BnW // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # 广播
            attn = attn.view(-1, self.num_heads, N, N)
        
        # Softmax和dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 注意力应用到V: (B*nW, num_heads, N, head_dim) -> (B*nW, N, C)
        x = (attn @ v).transpose(1, 2).reshape(BnW, N, C)
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# ============================================
# 3D Swin Transformer Block
# ============================================

class SwinTransformer3DBlock(nn.Module):
    """
    3D Swin Transformer Block
    
    这是3D Swin Transformer的基本构建块。
    每个block包含：
    1. Layer Norm
    2. Window Attention（或Shifted Window Attention）
    3. 残差连接
    4. Layer Norm
    5. MLP
    6. 残差连接
    
    参数:
        dim: 输入特征维度
        num_heads: 注意力头数
        window_size: 3D窗口大小
        shift_size: 移位大小（0表示不移位，即W-MSA；>0表示移位，即SW-MSA）
        mlp_ratio: MLP隐藏层维度与输入维度的比例
        qkv_bias: 是否使用偏置
        drop: Dropout率
        attn_drop: 注意力Dropout率
        drop_path: 随机深度率
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Layer Normalization 1（在注意力之前）
        self.norm1 = nn.LayerNorm(dim)
        
        # Window Attention模块
        self.attn = WindowAttention3D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Drop Path（随机深度）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Layer Normalization 2（在MLP之前）
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP（多层感知机）
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop
        )
    
    def forward(self, x, mask_matrix=None):
        """
        前向传播
        
        参数:
            x: 输入张量 (B, T, H, W, C)
            mask_matrix: 用于移位窗口的注意力掩膜
        
        返回:
            output: 输出张量 (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        shortcut = x  # 保存用于残差连接
        shift_size = get_effective_shift_size_3d((T, H, W), self.window_size, self.shift_size)
        
        # ===== 第1部分: Layer Norm + Window Attention + 残差连接 =====
        
        # Layer Normalization
        x = self.norm1(x)

        pad_t = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))

        Tp, Hp, Wp = x.shape[1:4]
        
        # 如果需要移位（shift_size > 0），则进行循环移位
        if any(s > 0 for s in shift_size):
            # 计算移位量（负号表示反向移位）
            shifted_x = torch.roll(
                x, 
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3)
            )
            # 使用提供的掩膜
            attn_mask = mask_matrix
        else:
            # 不移位
            shifted_x = x
            attn_mask = None
        
        # 窗口分区
        # (B, T, H, W, C) -> (num_windows*B, T_w*H_w*W_w, C)
        x_windows = window_partition_3d(shifted_x, self.window_size)
        
        # Window Attention
        # (num_windows*B, N, C) -> (num_windows*B, N, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # 窗口逆操作（合并窗口）
        # (num_windows*B, N, C) -> (B, T, H, W, C)
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, Tp, Hp, Wp)
        
        # 如果之前进行了移位，现在需要反向移位回来
        if any(s > 0 for s in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        if pad_t or pad_h or pad_w:
            x = x[:, :T, :H, :W, :].contiguous()
        
        # Drop Path + 残差连接
        x = shortcut + self.drop_path(x)
        
        # ===== 第2部分: Layer Norm + MLP + 残差连接 =====
        shortcut = x  # 再次保存
        
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        
        return x


# ============================================
# Patch Embedding (将输入分割成patches)
# ============================================

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding
    
    将输入的时空数据分割成3D patches，并投影到嵌入空间
    
    类似ViT中的patch embedding，但是扩展到3D（时间+空间）
    
    参数:
        patch_size: 3D patch大小 (T, H, W)
        in_chans: 输入通道数（SST数据通常是1）
        embed_dim: 嵌入维度（输出通道数）
    """
    
    def __init__(self, patch_size=(2, 4, 4), in_chans=1, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 使用3D卷积进行patch embedding
        # 步幅=stride=patch_size，这样每个patch被卷积成一个向量
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (B, C, T, H, W)
        
        返回:
            x: 输出张量 (B, embed_dim, T', H', W')，其中T'=T//patch_T等
        """
        B, C, T, H, W = x.shape
        
        pad_t = (self.patch_size[0] - T % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]

        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))

        # 3D卷积进行patch embedding
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        
        # 保存空间维度用于后续
        B, C, T_out, H_out, W_out = x.shape
        
        # Layer Norm（需要在通道维度上做）
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T', H', W', C)
        x = self.norm(x)
        
        # 恢复回 (B, C, T', H', W') 以便后续3D操作
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        return x


# ============================================
# Patch Merging (下采样)
# ============================================

class PatchMerging3D(nn.Module):
    """
    3D Patch Merging（下采样）
    
    在视频版Swin Transformer里，Patch Merging通常只对空间维度下采样，
    时间维度保持不变。这里也采用相同策略，避免时间长度在多阶段网络中
    过快缩小，导致window size与特征尺寸不兼容。
    
    这是实现层次化Transformer的关键！
    
    参数:
        dim: 输入特征维度
        out_dim: 输出特征维度（通常是dim的2倍）
    """
    
    def __init__(self, dim, out_dim=None):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        
        # 将空间上相邻的2x2个patch特征拼接后投影
        # 输入: 4 * dim（4个patch的特征拼接）
        # 输出: out_dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (B, C, T, H, W)
        
        返回:
            x: 输出张量 (B, out_dim, T, ceil(H/2), ceil(W/2))
        """
        B, C, T, H, W = x.shape

        # 先转成channels-last，方便按最后一个维度做LayerNorm/Linear
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, 0))

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]

        # (B, T, H/2, W/2, 4*C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        # Layer Norm + 降维投影
        x = self.norm(x)
        x = self.reduction(x)

        # 恢复回channels-first: (B, out_dim, T, H/2, W/2)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        return x


# ============================================
# 3D Swin Transformer Stage
# ============================================

class SwinTransformer3DStage(nn.Module):
    """
    3D Swin Transformer的一个阶段（Stage）
    
    每个Stage包含：
    1. 多个Swin Transformer Block（交替使用W-MSA和SW-MSA）
    2. 可选的Patch Merging（除最后一个stage外）
    
    参数:
        dim: 输入特征维度
        depth: 该stage中block的数量
        num_heads: 注意力头数
        window_size: 3D窗口大小
        mlp_ratio: MLP扩展比例
        qkv_bias: 是否使用QKV偏置
        drop: Dropout率
        attn_drop: 注意力Dropout率
        drop_path: 随机深度率（可以是列表，每个block不同）
        downsample: 是否在下采样（即是否是最后一个stage）
        out_dim: 下采样后的输出维度（如果downsample=True）
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 7, 7),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: List[float] = None,
        downsample: bool = False,
        out_dim: int = None
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        
        # 如果drop_path是单个数值，为每个block创建相同的值
        if drop_path is None:
            drop_path = [0.] * depth
        elif isinstance(drop_path, float):
            drop_path = [drop_path] * depth
        
        # 构建blocks
        # 注意：偶数索引（0, 2, 4...）使用W-MSA（不移位）
        #       奇数索引（1, 3, 5...）使用SW-MSA（移位）
        self.blocks = nn.ModuleList([
            SwinTransformer3DBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else tuple(w // 2 for w in window_size),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
            )
            for i in range(depth)
        ])
        
        # 下采样层（如果不是最后一个stage）
        if downsample:
            self.downsample = PatchMerging3D(dim=dim, out_dim=out_dim)
        else:
            self.downsample = None
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (B, C, T, H, W)
        
        返回:
            x: 输出张量
               如果downsample: (B, out_dim, T, H//2, W//2)
               否则: (B, C, T, H, W)
        """
        # Block内部使用channels-last布局，Stage入口和出口仍保持channels-first
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        for block in self.blocks:
            x_size = x.shape[1:4]
            shift_size = get_effective_shift_size_3d(x_size, block.window_size, block.shift_size)
            mask_matrix = compute_attention_mask_3d(
                x_size=x_size,
                window_size=block.window_size,
                shift_size=shift_size,
                device=x.device,
            )
            x = block(x, mask_matrix=mask_matrix)

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


# ============================================
# 完整的3D Swin Transformer模型
# ============================================

class SwinTransformer3D(nn.Module):
    """
    完整的3D Swin Transformer模型用于SST预测
    
    整体架构：
    1. Patch Embedding: 将输入分割成3D patches
    2. 多个Stage（每个Stage包含多个Swin Blocks + 可选的Patch Merging）
    3. 输出层：将特征映射回原始空间分辨率
    
    参数:
        patch_size: 3D patch大小 (T, H, W)
        in_chans: 输入通道数
        embed_dim: 初始嵌入维度
        depths: 每个stage的block数量列表，如 [2, 2, 6, 2]
        num_heads: 每个stage的注意力头数列表
        window_size: 3D窗口大小
        mlp_ratio: MLP扩展比例
        qkv_bias: 是否使用QKV偏置
        drop_rate: Dropout率
        attn_drop_rate: 注意力Dropout率
        drop_path_rate: 随机深度率（全局）
        output_dim: 输出维度（通常是1，表示SST）
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = (2, 7, 7),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        output_dim: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.num_stages = len(depths)
        self.output_dim = output_dim
        
        # ========================================
        # 1. Patch Embedding
        # ========================================
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # ========================================
        # 2. 计算随机深度衰减率（每个stage不同）
        # ========================================
        # 从0线性增加到drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # ========================================
        # 3. 构建各个Stage
        # ========================================
        self.stages = nn.ModuleList()
        
        for i_stage in range(self.num_stages):
            # 当前stage的参数
            stage_dim = int(embed_dim * (2 ** i_stage))  # 每个stage维度翻倍
            stage_depth = depths[i_stage]
            stage_num_heads = num_heads[i_stage]
            
            # 当前stage的随机深度率
            stage_dpr = dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])]
            
            # 是否下采样（除了第一个stage外，都在开始时包含下采样）
            # 实际上，我们在上一个stage的末尾已经下采样了
            # 所以这里除了第一个stage外，其他不需要再下采样
            downsample = (i_stage > 0)
            
            # 创建Stage
            stage = SwinTransformer3DStage(
                dim=stage_dim,
                depth=stage_depth,
                num_heads=stage_num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage_dpr,
                downsample=False,  # 我们在stage之间手动处理下采样
                out_dim=None
            )
            
            self.stages.append(stage)
            
            # 如果还有下一个stage，添加下采样层
            if i_stage < self.num_stages - 1:
                next_dim = int(embed_dim * (2 ** (i_stage + 1)))
                downsample_layer = PatchMerging3D(
                    dim=stage_dim,
                    out_dim=next_dim
                )
                self.stages.append(downsample_layer)
        
        # ========================================
        # 4. 输出层
        # ========================================
        # 最终的Layer Norm
        self.norm = nn.LayerNorm(int(embed_dim * (2 ** (self.num_stages - 1))))
        
        final_dim = int(embed_dim * (2 ** (self.num_stages - 1)))
        hidden_dim = max(final_dim // 2, 1)

        # 输出投影：先插值恢复到目标大小，再用可训练的1x1卷积映射到输出通道
        self.output_proj = nn.Sequential(
            nn.Conv3d(final_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 1, kernel_size=1),
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (B, C_in, T, H, W)
               B: batch size
               C_in: 输入通道（通常是1，表示SST）
               T: 时间步长（输入月份数）
               H, W: 空间高度和宽度
        
        返回:
            output: 输出张量 (B, C_out, T_out, H, W)
                    C_out: 输出通道（通常是1）
                    T_out: 输出时间步长（预测月份数，通常是1）
        """
        B, C_in, T, H, W = x.shape
        
        # 步骤1: Patch Embedding
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        
        # 步骤2: 通过各个Stage
        for layer in self.stages:
            x = layer(x)
        
        # 步骤3: 最终Layer Norm
        # 调整维度以适应LayerNorm: (B, C, T, H, W) -> (B, T, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        # 恢复: (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        # 步骤4: 恢复到目标时空分辨率
        x = F.interpolate(
            x, 
            size=(self.output_dim, H, W),
            mode='trilinear',
            align_corners=False
        )

        # 步骤5: 输出通道映射
        x = self.output_proj(x)
        
        return x


# ============================================
# 辅助函数和构建器
# ============================================

def build_swin_3d_tiny(**kwargs):
    """构建Swin-3D-Tiny模型"""
    config = {
        'patch_size': (2, 4, 4),
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
    }
    config.update(kwargs)
    return SwinTransformer3D(**config)


def build_swin_3d_small(**kwargs):
    """构建Swin-3D-Small模型"""
    config = {
        'patch_size': (2, 4, 4),
        'embed_dim': 96,
        'depths': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
    }
    config.update(kwargs)
    return SwinTransformer3D(**config)


def build_swin_3d_base(**kwargs):
    """构建Swin-3D-Base模型"""
    config = {
        'patch_size': (2, 4, 4),
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
    }
    config.update(kwargs)
    return SwinTransformer3D(**config)


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 3D Swin Transformer 模型")
    print("=" * 70)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建一个小型模型用于测试
    print("\n创建Swin-3D-Tiny模型...")
    model = build_swin_3d_tiny(
        window_size=(2, 7, 7),
        output_dim=1
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - 可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    
    # 创建测试输入
    batch_size = 2
    in_channels = 1
    input_months = 12
    height = 56  # 使用较小的尺寸以便测试（原始可能是180）
    width = 56
    
    x = torch.randn(batch_size, in_channels, input_months, height, width).to(device)
    model = model.to(device)
    model.eval()
    
    print(f"\n输入形状: {x.shape}")
    print(f"  - Batch size: {x.shape[0]}")
    print(f"  - 通道数: {x.shape[1]}")
    print(f"  - 时间步: {x.shape[2]}")
    print(f"  - 空间尺寸: {x.shape[3]} x {x.shape[4]}")
    
    # 前向传播（不计算梯度）
    with torch.no_grad():
        try:
            output = model(x)
            
            print(f"\n输出形状: {output.shape}")
            print(f"  - Batch size: {output.shape[0]}")
            print(f"  - 通道数: {output.shape[1]}")
            print(f"  - 时间步: {output.shape[2]}")
            print(f"  - 空间尺寸: {output.shape[3]} x {output.shape[4]}")
            
            # 验证输出范围
            print(f"\n输出统计:")
            print(f"  - 最小值: {output.min().item():.4f}")
            print(f"  - 最大值: {output.max().item():.4f}")
            print(f"  - 均值: {output.mean().item():.4f}")
            
            print("\n" + "=" * 70)
            print("前向传播测试成功！")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试内存占用
    if torch.cuda.is_available():
        print("\n内存统计:")
        print(f"  - 分配显存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  - 保留显存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
