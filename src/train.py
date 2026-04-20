"""
训练脚本 - 用于训练3D Swin Transformer模型
=============================================

这个脚本负责：
1. 加载配置（命令行参数 + 配置文件）
2. 创建数据加载器
3. 创建模型
4. 设置优化器和学习率调度器
5. 训练循环（训练 + 验证）
6. 保存检查点
7. 记录日志（TensorBoard）

新手提示：
- 这是典型的深度学习训练流程
- 建议先理解每个模块的作用，再运行
- 如果显存不足，请减小batch_size或spatial_downsample
"""

import os
import sys
import argparse
import yaml
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import create_data_loaders, SSTDataProcessor
from model_3dswin import (
    SwinTransformer3D,
    build_swin_3d_tiny,
    build_swin_3d_small,
    build_swin_3d_base
)


class NullSummaryWriter:
    """当TensorBoard不可用时使用的空writer。"""

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def parse_args():
    """
    解析命令行参数
    
    可以通过两种方式指定参数：
    1. 命令行直接指定，如：--batch_size 4 --epochs 50
    2. 通过配置文件，如：--config configs/my_config.yaml
    
    命令行参数的优先级高于配置文件
    """
    parser = argparse.ArgumentParser(
        description='Train 3D Swin Transformer for SST Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ==========================================
    # 配置文件
    # ==========================================
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径 (YAML格式)'
    )
    
    # ==========================================
    # 数据相关参数
    # ==========================================
    data_group = parser.add_argument_group('Data Arguments', '数据加载相关参数')
    
    data_group.add_argument(
        '--data_path',
        type=str,
        default='HadISST_sst.nc',
        help='SST数据文件路径'
    )
    
    data_group.add_argument(
        '--input_months',
        type=int,
        default=12,
        help='输入时间序列长度（过去多少个月）'
    )
    
    data_group.add_argument(
        '--output_months',
        type=int,
        default=1,
        help='输出时间序列长度（预测多少个月）'
    )
    
    data_group.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='训练批次大小'
    )
    
    data_group.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='数据加载线程数（Windows建议0）'
    )
    
    data_group.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='训练集比例'
    )
    
    data_group.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='验证集比例（测试集 = 1 - train_ratio - val_ratio）'
    )
    
    data_group.add_argument(
        '--normalize_method',
        type=str,
        default='zscore',
        choices=['zscore', 'minmax', 'none'],
        help='数据归一化方法'
    )
    
    data_group.add_argument(
        '--spatial_downsample',
        type=int,
        default=1,
        help='空间降采样因子（1=不降采样，2=减半，4=四分之一）'
    )
    
    # ==========================================
    # 模型相关参数
    # ==========================================
    model_group = parser.add_argument_group('Model Arguments', '模型架构相关参数')
    
    model_group.add_argument(
        '--model_type',
        type=str,
        default='tiny',
        choices=['tiny', 'small', 'base', 'custom'],
        help='预定义的模型类型（tiny/small/base）或custom自定义'
    )
    
    model_group.add_argument(
        '--patch_size',
        type=int,
        nargs=3,
        default=[2, 4, 4],
        help='3D Patch大小 (T, H, W)'
    )
    
    model_group.add_argument(
        '--embed_dim',
        type=int,
        default=96,
        help='初始嵌入维度'
    )
    
    model_group.add_argument(
        '--depths',
        type=int,
        nargs='+',
        default=[2, 2, 6, 2],
        help='每个stage的深度（block数量）'
    )
    
    model_group.add_argument(
        '--num_heads',
        type=int,
        nargs='+',
        default=[3, 6, 12, 24],
        help='每个stage的注意力头数'
    )
    
    model_group.add_argument(
        '--window_size',
        type=int,
        nargs=3,
        default=[2, 7, 7],
        help='3D窗口大小 (T, H, W)'
    )
    
    model_group.add_argument(
        '--mlp_ratio',
        type=float,
        default=4.0,
        help='MLP隐藏层维度与输入维度的比例'
    )
    
    model_group.add_argument(
        '--drop_rate',
        type=float,
        default=0.0,
        help='Dropout率'
    )
    
    model_group.add_argument(
        '--attn_drop_rate',
        type=float,
        default=0.0,
        help='注意力Dropout率'
    )
    
    model_group.add_argument(
        '--drop_path_rate',
        type=float,
        default=0.1,
        help='随机深度率（全局）'
    )
    
    # ==========================================
    # 训练相关参数
    # ==========================================
    train_group = parser.add_argument_group('Training Arguments', '训练相关参数')
    
    train_group.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='训练轮数'
    )
    
    train_group.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='学习率'
    )
    
    train_group.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='权重衰减（L2正则化）'
    )
    
    train_group.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        choices=['adam', 'adamw', 'sgd'],
        help='优化器类型'
    )
    
    train_group.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        choices=['step', 'cosine', 'plateau', 'none'],
        help='学习率调度器类型'
    )
    
    # ==========================================
    # 日志和保存相关参数
    # ==========================================
    log_group = parser.add_argument_group('Logging and Saving Arguments')
    
    log_group.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='输出目录（用于保存日志、检查点等）'
    )
    
    log_group.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='实验名称（用于创建子目录）'
    )
    
    log_group.add_argument(
        '--save_freq',
        type=int,
        default=5,
        help='每隔多少轮保存一次检查点'
    )
    
    log_group.add_argument(
        '--log_freq',
        type=int,
        default=10,
        help='每隔多少批次记录一次日志'
    )
    
    log_group.add_argument(
        '--eval_freq',
        type=int,
        default=1,
        help='每隔多少轮验证一次'
    )
    
    # ==========================================
    # 其他参数
    # ==========================================
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（为了结果可复现）'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从检查点恢复训练（检查点文件路径）'
    )
    
    parser.add_argument(
        '--test_only',
        action='store_true',
        help='仅测试模式（不训练）'
    )
    
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='使用混合精度训练（FP16），可以加速训练并减少显存占用'
    )
    
    args = parser.parse_args()
    args._parser_defaults = {
        action.dest: parser.get_default(action.dest)
        for action in parser._actions
        if action.dest != 'help'
    }
    
    return args


def load_config(args):
    """
    加载配置文件并与命令行参数合并
    
    优先级：命令行参数 > 配置文件 > 默认值
    
    参数:
        args: 解析后的命令行参数
    
    返回:
        更新后的参数
    """
    config_path = Path(args.config)
    
    if config_path.exists():
        print(f"\n加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 将配置文件中的参数更新到args（如果命令行没有指定）
        for key, value in config.items():
            if hasattr(args, key):
                # 检查命令行是否使用了默认值
                current_value = getattr(args, key)
                parser_default = args._parser_defaults.get(key, current_value)
                
                # 如果当前值是默认值，则使用配置文件中的值
                if current_value == parser_default:
                    setattr(args, key, value)
                    print(f"  从配置加载: {key} = {value}")
            else:
                # 如果args没有这个属性，直接添加
                setattr(args, key, value)
                print(f"  从配置加载: {key} = {value}")
    else:
        print(f"\n配置文件不存在: {config_path}")
        print("将使用默认参数和命令行参数")
    
    return args


def setup_logger(args):
    """
    设置日志记录器
    
    返回:
        output_dir: 输出目录路径
        writer: TensorBoard writer
    """
    # 创建实验目录
    if args.exp_name is None:
        # 自动生成实验名称：日期_时间
        args.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # 创建TensorBoard writer
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(log_dir))
    else:
        writer = NullSummaryWriter()
        print("  - TensorBoard不可用：将跳过日志写入（可安装 tensorboard 后启用）")
    
    # 保存配置
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        # 将args转换为字典并保存
        args_dict = {k: v for k, v in vars(args).items() if k != '_parser_defaults'}
        yaml.dump(args_dict, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n日志设置完成:")
    print(f"  - 实验名称: {args.exp_name}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 检查点目录: {checkpoint_dir}")
    print(f"  - TensorBoard日志: {log_dir}")
    print(f"  - 配置文件已保存: {config_save_path}")
    
    return output_dir, writer


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    args,
    writer,
    global_step
):
    """
    训练一个epoch
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler: 混合精度scaler
        device: 设备
        epoch: 当前epoch
        args: 参数
        writer: TensorBoard writer
        global_step: 全局步数
    
    返回:
        avg_loss: 平均损失
        global_step: 更新后的全局步数
    """
    model.train()
    
    total_loss = 0.0
    num_batches = len(train_loader)
    
    start_time = time.time()
    
    for batch_idx, (X, y, mask) in enumerate(train_loader):
        # 移动数据到设备
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 混合精度训练
        if args.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                # 前向传播
                output = model(X)
                
                # 计算损失（只在有效像素上计算）
                # output: (B, C_out, T_out, H, W)
                # y: (B, C_out, T_out, H, W)
                # mask: (B, C_out, T_out, H, W) - bool类型
                
                # 展平以便计算
                output_flat = output.view(-1)
                y_flat = y.view(-1)
                mask_flat = mask.view(-1).bool()
                valid_idx = mask_flat & torch.isfinite(output_flat) & torch.isfinite(y_flat)
                
                # 只在有效像素上计算损失
                if valid_idx.any():
                    loss = criterion(output_flat[valid_idx], y_flat[valid_idx])
                else:
                    # 如果没有有效像素，跳过这个batch
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            
        else:
            # 普通训练（不用混合精度）
            # 前向传播
            output = model(X)
            
            # 计算损失
            output_flat = output.view(-1)
            y_flat = y.view(-1)
            mask_flat = mask.view(-1).bool()
            valid_idx = mask_flat & torch.isfinite(output_flat) & torch.isfinite(y_flat)
            
            if valid_idx.any():
                loss = criterion(output_flat[valid_idx], y_flat[valid_idx])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        global_step += 1
        
        # 记录到TensorBoard（每隔一定步数）
        if global_step % args.log_freq == 0:
            writer.add_scalar('Train/Loss_step', loss.item(), global_step)
            writer.add_scalar('Train/Learning_rate', optimizer.param_groups[0]['lr'], global_step)
        
        # 打印进度
        if batch_idx % args.log_freq == 0 or batch_idx == num_batches - 1:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (batch_idx + 1) * (num_batches - batch_idx - 1)
            
            print(f"\rEpoch [{epoch}/{args.epochs}] "
                  f"Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.6f} "
                  f"Avg: {avg_loss:.6f} "
                  f"ETA: {eta:.0f}s",
                  end='', flush=True)
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    
    print(f"\nEpoch [{epoch}/{args.epochs}] 完成，平均损失: {avg_loss:.6f}")
    
    return avg_loss, global_step


def validate(model, val_loader, criterion, device, epoch, args, writer):
    """
    验证模型
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        args: 参数
        writer: TensorBoard writer
    
    返回:
        avg_loss: 平均验证损失
        avg_rmse: 平均RMSE
    """
    model.eval()
    
    total_loss = 0.0
    total_rmse = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (X, y, mask) in enumerate(val_loader):
            # 移动数据到设备
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            # 前向传播
            output = model(X)
            
            # 计算损失
            output_flat = output.view(-1)
            y_flat = y.view(-1)
            mask_flat = mask.view(-1).bool()
            valid_idx = mask_flat & torch.isfinite(output_flat) & torch.isfinite(y_flat)
            
            if valid_idx.any():
                loss = criterion(output_flat[valid_idx], y_flat[valid_idx])
                
                # 计算RMSE
                mse = F.mse_loss(output_flat[valid_idx], y_flat[valid_idx])
                rmse = torch.sqrt(mse)
            else:
                loss = torch.tensor(0.0, device=device)
                rmse = torch.tensor(0.0, device=device)
            
            total_loss += loss.item()
            total_rmse += rmse.item()
    
    # 计算平均
    avg_loss = total_loss / num_batches
    avg_rmse = total_rmse / num_batches
    
    # 记录到TensorBoard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/RMSE', avg_rmse, epoch)
    
    print(f"\n验证 - Epoch [{epoch}] "
          f"平均损失: {avg_loss:.6f}, "
          f"RMSE: {avg_rmse:.6f}")
    
    return avg_loss, avg_rmse


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """
    保存检查点
    
    参数:
        state: 要保存的状态字典
        is_best: 是否是最佳模型
        checkpoint_dir: 检查点保存目录
        filename: 文件名
    """
    checkpoint_path = Path(checkpoint_dir) / filename
    torch.save(state, checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = Path(checkpoint_dir) / 'model_best.pth'
        torch.save(state, best_path)
        print(f"最佳模型已保存: {best_path}")


def main():
    """
    主函数
    """
    # ==========================================
    # 步骤1: 解析参数
    # ==========================================
    args = parse_args()
    
    # 加载配置文件（如果有）
    args = load_config(args)
    
    # 设置随机种子（为了结果可复现）
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # ==========================================
    # 步骤2: 设置设备
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 70)
    print("设备信息")
    print("=" * 70)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 70)
    
    # ==========================================
    # 步骤3: 创建数据加载器
    # ==========================================
    print("\n" + "=" * 70)
    print("创建数据加载器")
    print("=" * 70)
    
    data_result = create_data_loaders(
        data_path=args.data_path,
        input_months=args.input_months,
        output_months=args.output_months,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        normalize_method=args.normalize_method,
        spatial_downsample=args.spatial_downsample
    )
    
    train_loader = data_result['train_loader']
    val_loader = data_result['val_loader']
    test_loader = data_result['test_loader']
    processor = data_result['processor']
    
    print("=" * 70)
    
    # ==========================================
    # 步骤4: 创建模型
    # ==========================================
    print("\n" + "=" * 70)
    print("创建模型")
    print("=" * 70)
    
    if args.model_type == 'tiny':
        model = build_swin_3d_tiny(
            patch_size=tuple(args.patch_size),
            in_chans=1,
            window_size=tuple(args.window_size),
            output_dim=args.output_months,
            drop_path_rate=args.drop_path_rate
        )
    elif args.model_type == 'small':
        model = build_swin_3d_small(
            patch_size=tuple(args.patch_size),
            in_chans=1,
            window_size=tuple(args.window_size),
            output_dim=args.output_months,
            drop_path_rate=args.drop_path_rate
        )
    elif args.model_type == 'base':
        model = build_swin_3d_base(
            patch_size=tuple(args.patch_size),
            in_chans=1,
            window_size=tuple(args.window_size),
            output_dim=args.output_months,
            drop_path_rate=args.drop_path_rate
        )
    else:  # custom
        model = SwinTransformer3D(
            patch_size=tuple(args.patch_size),
            in_chans=1,
            embed_dim=args.embed_dim,
            depths=args.depths,
            num_heads=args.num_heads,
            window_size=tuple(args.window_size),
            mlp_ratio=args.mlp_ratio,
            qkv_bias=args.qkv_bias,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path_rate,
            output_dim=args.output_months
        )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型信息:")
    print(f"  - 模型类型: {args.model_type}")
    print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - 可训练参数量: {trainable_params:,}")
    
    # 移动模型到设备
    model = model.to(device)
    
    print("=" * 70)
    
    # 如果只是测试模型，到这里就结束了
    if args.test_only:
        print("\n测试模式：只测试模型前向传播")
        model.eval()
        with torch.no_grad():
            # 获取一个测试batch
            X_test, y_test, mask_test = next(iter(train_loader))
            X_test = X_test.to(device)
            
            print(f"\n输入形状: {X_test.shape}")
            output = model(X_test)
            print(f"输出形状: {output.shape}")
            print("\n测试成功！")
        return

    output_dir, writer = setup_logger(args)
    
    # ==========================================
    # 步骤5: 设置优化器和学习率调度器
    # ==========================================
    print("\n" + "=" * 70)
    print("设置优化器和调度器")
    print("=" * 70)
    
    # 优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    print(f"优化器: {args.optimizer}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 权重衰减: {args.weight_decay}")
    
    # 学习率调度器
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:  # none
        scheduler = None
    
    if scheduler is not None:
        print(f"学习率调度器: {args.scheduler}")
    else:
        print("学习率调度器: 无（固定学习率）")
    
    print("=" * 70)
    
    # ==========================================
    # 步骤6: 设置损失函数和混合精度训练
    # ==========================================
    print("\n设置损失函数和训练工具")
    
    # 损失函数 - 使用MSE Loss（回归任务）
    criterion = nn.MSELoss(reduction='mean')
    
    # 混合精度训练的scaler
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("启用混合精度训练 (FP16)")
    
    print("=" * 70)
    
    # ==========================================
    # 步骤7: 开始训练
    # ==========================================
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    best_val_loss = float('inf')
    best_val_rmse = float('inf')
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"{'='*70}")
        
        # 训练一个epoch
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            args,
            writer,
            global_step
        )
        
        # 记录训练损失
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        
        # 验证
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            val_loss, val_rmse = validate(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                args,
                writer
            )
            
            writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            writer.add_scalar('Epoch/Val_RMSE', val_rmse, epoch)
            
            # 学习率调度（ReduceLROnPlateau）
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_val_rmse = val_rmse
                print(f"\n*** 新的最佳验证损失: {best_val_loss:.6f} ***")
            
            # 保存检查点
            if epoch % args.save_freq == 0 or is_best or epoch == args.epochs:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_rmse': best_val_rmse,
                    'args': args,
                    'global_step': global_step
                }
                
                if scaler is not None:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                
                save_checkpoint(
                    checkpoint,
                    is_best,
                    checkpoint_dir=Path(args.output_dir) / args.exp_name / 'checkpoints',
                    filename=f'checkpoint_epoch_{epoch}.pth'
                )
        
        # 其他学习率调度
        if scheduler is not None and args.scheduler != 'plateau':
            scheduler.step()
        
        # 记录epoch时间
        epoch_time = time.time() - epoch_start_time
        writer.add_scalar('Epoch/Time', epoch_time, epoch)
        
        print(f"\nEpoch [{epoch}] 完成，耗时: {epoch_time:.2f}s")
        print(f"当前最佳验证损失: {best_val_loss:.6f}, RMSE: {best_val_rmse:.6f}")
    
    # ==========================================
    # 训练完成
    # ==========================================
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"总训练轮数: {args.epochs}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳验证RMSE: {best_val_rmse:.6f}")
    print(f"\n输出目录: {Path(args.output_dir) / args.exp_name}")
    print("=" * 70)
    
    # 关闭TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
