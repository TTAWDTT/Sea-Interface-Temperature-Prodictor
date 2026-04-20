"""
预测脚本 - 使用训练好的模型进行SST预测
=========================================

这个脚本负责：
1. 加载训练好的模型检查点
2. 加载测试数据
3. 进行预测
4. 评估预测结果
5. 可视化预测结果
6. 保存预测结果

新手提示：
- 这是训练完成后的推理阶段
- 可以独立运行，不需要重新训练
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import create_data_loaders, SSTDataProcessor, save_predictions
from model_3dswin import SwinTransformer3D
from utils import (
    set_seed, get_device, print_model_info,
    calculate_rmse, calculate_mae, calculate_acc, evaluate_model,
    visualize_sst_map, plot_prediction_comparison, save_model_info
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Predict SST using trained 3D Swin Transformer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型检查点
    parser.add_argument(
        '--checkpoint',
        '-c',
        type=str,
        required=True,
        help='训练好的模型检查点路径（.pth文件）'
    )
    
    # 数据参数
    parser.add_argument(
        '--data_path',
        type=str,
        default='HadISST_sst.nc',
        help='SST数据文件路径'
    )
    
    # 输出参数
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='输出目录'
    )
    
    parser.add_argument(
        '--exp_name',
        type=str,
        default='prediction',
        help='实验名称'
    )
    
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='是否保存预测结果为NetCDF文件'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='是否可视化预测结果'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='可视化的样本数量'
    )

    parser.add_argument(
        '--rollout_months',
        type=int,
        default=0,
        help='滚动预测月数（>0时启用自回归滚动预测）'
    )

    parser.add_argument(
        '--rollout_start_idx',
        type=int,
        default=0,
        help='滚动预测起始样本索引（基于所选数据集）'
    )

    parser.add_argument(
        '--rollout_split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='滚动预测使用的数据集划分'
    )

    parser.add_argument(
        '--direct_checkpoints',
        nargs='+',
        default=None,
        help='用于对比的直接多步预测checkpoint列表（例如 2/4/6/12个月模型）'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='批次大小'
    )
    
    parser.add_argument(
        '--spatial_downsample',
        type=int,
        default=4,
        help='空间降采样因子（应与训练时一致）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='计算设备（cuda/cpu，默认自动检测）'
    )
    
    args = parser.parse_args()
    return args


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    从检查点加载模型
    
    参数:
        checkpoint_path: 检查点文件路径
        device: 计算设备
    
    返回:
        model: 加载好的模型
        checkpoint: 检查点字典
    """
    print(f"\n加载模型检查点: {checkpoint_path}")
    
    # 加载检查点
    # PyTorch 2.6 起 torch.load 默认 weights_only=True，
    # 本项目checkpoint包含 argparse.Namespace 等对象，因此需要显式关闭。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型配置
    args = checkpoint['args']
    
    # 创建模型
    model = SwinTransformer3D(
        patch_size=tuple(args.patch_size) if hasattr(args, 'patch_size') else (2, 4, 4),
        in_chans=1,
        embed_dim=args.embed_dim if hasattr(args, 'embed_dim') else 96,
        depths=args.depths if hasattr(args, 'depths') else [2, 2, 6, 2],
        num_heads=args.num_heads if hasattr(args, 'num_heads') else [3, 6, 12, 24],
        window_size=tuple(args.window_size) if hasattr(args, 'window_size') else (2, 7, 7),
        mlp_ratio=args.mlp_ratio if hasattr(args, 'mlp_ratio') else 4.0,
        drop_path_rate=args.drop_path_rate if hasattr(args, 'drop_path_rate') else 0.1,
        output_dim=args.output_months if hasattr(args, 'output_months') else 1
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 移动模型到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    print("模型加载完成！")
    print(f"  - 训练轮次: {checkpoint['epoch']}")
    print(f"  - 最佳验证损失: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"  - 最佳验证RMSE: {checkpoint.get('best_val_rmse', 'N/A')}")
    
    return model, checkpoint


def load_checkpoint_metadata(checkpoint_path: str):
    """只读取checkpoint元信息，用于对比模式预检查。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    return checkpoint, args


def predict_on_dataset(model, dataloader, device, processor):
    """
    在整个数据集上进行预测
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        processor: 数据处理器（用于反归一化）
    
    返回:
        all_predictions: 所有预测值
        all_targets: 所有真实值
        all_masks: 所有掩膜
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch_idx, (X, y, mask) in enumerate(dataloader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            # 前向传播
            pred = model(X)
            
            # 收集结果
            all_predictions.append(pred.cpu())
            all_targets.append(y.cpu())
            all_masks.append(mask.cpu())
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(f"\r预测进度: [{batch_idx+1}/{len(dataloader)}]", end='', flush=True)
    
    print()  # 换行
    
    # 拼接所有结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # 反归一化
    all_predictions_np = all_predictions.numpy()
    all_targets_np = all_targets.numpy()
    
    # 注意：processor的denormalize方法需要特定shape，这里我们手动反归一化
    if hasattr(processor, 'stats') and processor.stats['mean'] is not None:
        mean = processor.stats['mean']
        std = processor.stats['std']
        
        all_predictions_np = all_predictions_np * std + mean
        all_targets_np = all_targets_np * std + mean
    
    return all_predictions_np, all_targets_np, all_masks.numpy()


def save_predictions_to_netcdf(predictions, save_path, dates=None, lat=None, lon=None):
    """
    保存预测结果为NetCDF格式
    
    参数:
        predictions: 预测结果数组 (N, C, H, W) 或 (N, C, T, H, W)
        save_path: 保存路径
        dates: 日期数组（可选）
        lat: 纬度数组（可选）
        lon: 经度数组（可选）
    """
    # 支持输入形状:
    # 1) (N, C, T, H, W)
    # 2) (N, T, H, W)
    # 3) (N, H, W)
    # 4) (T, H, W) - 如rollout结果
    if predictions.ndim == 5:
        # 单通道任务，去掉C维
        if predictions.shape[1] == 1:
            predictions = predictions[:, 0]  # -> (N, T, H, W)
        else:
            raise ValueError(f"不支持多通道预测保存，当前shape={predictions.shape}")

    if predictions.ndim == 4:
        # 约定为 (N, T, H, W)
        N, T, H, W = predictions.shape
        if dates is None:
            dates = np.arange(N)
        if lat is None:
            lat = np.arange(H)
        if lon is None:
            lon = np.arange(W)

        coords = [dates, np.arange(T), lat, lon]
        dims = ['sample', 'time', 'latitude', 'longitude']

    elif predictions.ndim == 3:
        # 兼容 (N, H, W) 或 rollout 的 (T, H, W)
        N, H, W = predictions.shape
        if dates is None:
            dates = np.arange(N)
        if lat is None:
            lat = np.arange(H)
        if lon is None:
            lon = np.arange(W)

        coords = [dates, lat, lon]
        dims = ['sample', 'latitude', 'longitude']

    else:
        raise ValueError(f"不支持的预测数组维度: {predictions.ndim}, shape={predictions.shape}")
    
    # 创建DataArray
    da = xr.DataArray(
        predictions,
        coords=coords,
        dims=dims,
        name='sst_prediction',
        attrs={
            'units': 'degrees_Celsius',
            'long_name': 'Sea Surface Temperature Prediction',
            'description': 'Predicted by 3D Swin Transformer'
        }
    )
    
    # 保存为NetCDF
    ds = da.to_dataset(name='sst')
    ds.to_netcdf(save_path)
    print(f"预测结果已保存: {save_path}")


def rollout_forecast(model, init_input, months, device):
    """
    自回归滚动预测

    参数:
        model: 训练好的模型
        init_input: 初始输入，形状 (1, 1, T_in, H, W)
        months: 需要滚动预测的月份数
        device: 设备

    返回:
        rollout_pred: 预测结果，形状 (1, 1, months, H, W)
    """
    model.eval()

    current = init_input.to(device)
    preds = []

    with torch.no_grad():
        for _ in range(months):
            out = model(current)  # (1, 1, T_out, H, W)

            # 使用第一个预测步进行自回归，兼容output_months=1和>1的模型
            next_step = out[:, :, 0:1, :, :]
            preds.append(next_step.cpu())

            # 滑动窗口：丢弃最早1个月，拼接最新预测1个月
            current = torch.cat([current[:, :, 1:, :, :], next_step.to(device)], dim=2)

    rollout_pred = torch.cat(preds, dim=2)  # (1, 1, months, H, W)
    return rollout_pred


def rollout_forecast_batch(model, init_input, months, device, oracle_targets=None):
    """
    对一个batch做滚动预测；如果提供oracle_targets，则后续步使用真实值驱动。

    参数:
        model: 训练好的模型
        init_input: 输入张量 (B, 1, T_in, H, W)
        months: 滚动步数
        device: 设备
        oracle_targets: 可选真实序列 (B, 1, months, H, W)

    返回:
        rollout_pred: (B, 1, months, H, W)
    """
    model.eval()
    current = init_input.to(device)
    preds = []

    with torch.no_grad():
        for step in range(months):
            out = model(current)
            next_step = out[:, :, 0:1, :, :]
            preds.append(next_step.cpu())

            if oracle_targets is not None:
                next_input = oracle_targets[:, :, step:step + 1, :, :].to(device)
            else:
                next_input = next_step.to(device)

            current = torch.cat([current[:, :, 1:, :, :], next_input], dim=2)

    return torch.cat(preds, dim=2)


def oracle_rollout_forecast(model, init_input, future_truth_steps, device):
    """
    使用真实未来值作为下一步输入的 oracle rollout。

    参数:
        model: 训练好的模型
        init_input: 初始输入，形状 (1, 1, T_in, H, W)
        future_truth_steps: 真实未来步，形状 (1, 1, months, H, W)
        device: 设备

    返回:
        oracle_pred: 预测结果，形状 (1, 1, months, H, W)
    """
    model.eval()

    current = init_input.to(device)
    preds = []
    months = future_truth_steps.shape[2]

    with torch.no_grad():
        for step_idx in range(months):
            out = model(current)
            next_pred = out[:, :, 0:1, :, :]
            preds.append(next_pred.cpu())

            truth_step = future_truth_steps[:, :, step_idx:step_idx + 1, :, :].to(device)
            current = torch.cat([current[:, :, 1:, :, :], truth_step], dim=2)

    oracle_pred = torch.cat(preds, dim=2)
    return oracle_pred


def denormalize_numpy(data_np, processor):
    """按数据处理器统计量反归一化numpy数组。"""
    if hasattr(processor, 'stats') and processor.stats['mean'] is not None:
        mean = processor.stats['mean']
        std = processor.stats['std']
        return data_np * std + mean
    return data_np


def extract_horizon_sample(dataset, start_idx, horizon):
    """取滚动比较中第 horizon 个提前期对应的真实样本。"""
    sample_idx = start_idx + horizon - 1
    if sample_idx >= len(dataset):
        raise IndexError(f"样本索引越界: start_idx={start_idx}, horizon={horizon}, dataset_len={len(dataset)}")
    return dataset[sample_idx]


def build_future_truth_steps(dataset, start_idx, months):
    """构建 oracle rollout 需要的连续真实未来步。"""
    truth_steps = []
    for offset in range(months):
        _, y_step, _ = dataset[start_idx + offset]
        truth_steps.append(y_step.unsqueeze(2))
    return torch.cat(truth_steps, dim=2)


def compute_metrics_np(pred, target, mask=None):
    """在numpy数组上计算RMSE/MAE/ACC。"""
    if mask is None:
        valid = np.isfinite(pred) & np.isfinite(target)
    else:
        valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)

    if not np.any(valid):
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'acc': float('nan'),
            'valid_points': 0
        }

    p = pred[valid]
    t = target[valid]

    rmse = np.sqrt(np.mean((p - t) ** 2))
    mae = np.mean(np.abs(p - t))

    p_anom = p - np.mean(p)
    t_anom = t - np.mean(t)
    denom = np.sqrt(np.sum(p_anom ** 2)) * np.sqrt(np.sum(t_anom ** 2))
    if denom <= 1e-12:
        acc = 0.0
    else:
        acc = np.sum(p_anom * t_anom) / denom

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'acc': float(acc),
        'valid_points': int(valid.sum())
    }


def save_comparison_plot(df, save_path):
    """保存各方法在不同horizon上的RMSE/MAE/ACC对比图。"""
    if df.empty:
        return

    horizons = sorted(df['horizon'].unique())
    methods = ['direct', 'rollout', 'oracle']
    colors = {'direct': 'tab:blue', 'rollout': 'tab:orange', 'oracle': 'tab:green'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for method in methods:
        sub = df[df['method'] == method].sort_values('horizon')
        if sub.empty:
            continue
        axes[0].plot(sub['horizon'], sub['rmse'], marker='o', linewidth=2, label=method, color=colors[method])
        axes[1].plot(sub['horizon'], sub['mae'], marker='o', linewidth=2, label=method, color=colors[method])
        axes[2].plot(sub['horizon'], sub['acc'], marker='o', linewidth=2, label=method, color=colors[method])

    axes[0].set_title('RMSE Comparison')
    axes[0].set_xlabel('Horizon (months)')
    axes[0].set_ylabel('RMSE (degC)')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('MAE Comparison')
    axes[1].set_xlabel('Horizon (months)')
    axes[1].set_ylabel('MAE (degC)')
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title('ACC Comparison')
    axes[2].set_xlabel('Horizon (months)')
    axes[2].set_ylabel('ACC')
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].grid(True, alpha=0.3)

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def evaluate_last_step_metrics(pred_seq, target_seq, mask_seq, horizon):
    """只评估第horizon个月（1-based）的指标。"""
    pred_last = pred_seq[:, :, horizon - 1, :, :].cpu().numpy()
    target_last = target_seq[:, :, horizon - 1, :, :].cpu().numpy()
    mask_last = mask_seq[:, :, horizon - 1, :, :].cpu().numpy()
    return compute_metrics_np(pred_last, target_last, mask_last)


def compare_horizon_methods(base_model, direct_checkpoints, dataloader, device, output_dir, processor):
    """比较直接多步模型、自由滚动和oracle滚动在不同horizon上的表现。"""
    comparison_rows = []

    # 先加载直接模型和其对应horizon
    direct_entries = []
    for ckpt_path in direct_checkpoints:
        direct_model, direct_ckpt = load_model_from_checkpoint(ckpt_path, device)
        horizon = int(getattr(direct_ckpt['args'], 'output_months', 1))
        direct_entries.append((horizon, ckpt_path, direct_model))

    direct_entries.sort(key=lambda x: x[0])

    for horizon, ckpt_path, direct_model in direct_entries:
        print(f"\n评估 horizon={horizon} 的直接模型: {ckpt_path}")

        direct_predictions = []
        rollout_predictions = []
        oracle_predictions = []
        targets_list = []
        masks_list = []

        with torch.no_grad():
            for batch_idx, (X, y, mask) in enumerate(dataloader):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                direct_pred = direct_model(X)
                rollout_pred = rollout_forecast_batch(base_model, X, horizon, device)
                oracle_pred = rollout_forecast_batch(base_model, X, horizon, device, oracle_targets=y)

                direct_predictions.append(direct_pred[:, :, horizon - 1].cpu())
                rollout_predictions.append(rollout_pred[:, :, horizon - 1].cpu())
                oracle_predictions.append(oracle_pred[:, :, horizon - 1].cpu())
                targets_list.append(y[:, :, horizon - 1].cpu())
                masks_list.append(mask[:, :, horizon - 1].cpu())

                if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:
                    print(f"\r比较进度: [{batch_idx+1}/{len(dataloader)}]", end='', flush=True)

        print()

        direct_pred_np = denormalize_numpy(torch.cat(direct_predictions, dim=0).numpy(), processor)
        rollout_pred_np = denormalize_numpy(torch.cat(rollout_predictions, dim=0).numpy(), processor)
        oracle_pred_np = denormalize_numpy(torch.cat(oracle_predictions, dim=0).numpy(), processor)
        target_np = denormalize_numpy(torch.cat(targets_list, dim=0).numpy(), processor)
        mask_np = torch.cat(masks_list, dim=0).numpy()

        direct_metrics = compute_metrics_np(direct_pred_np, target_np, mask_np)
        rollout_metrics = compute_metrics_np(rollout_pred_np, target_np, mask_np)
        oracle_metrics = compute_metrics_np(oracle_pred_np, target_np, mask_np)

        comparison_rows.extend([
            {
                'method': 'direct',
                'horizon': horizon,
                'checkpoint': ckpt_path,
                **direct_metrics,
            },
            {
                'method': 'rollout',
                'horizon': horizon,
                'checkpoint': str(base_model.__class__.__name__),
                **rollout_metrics,
            },
            {
                'method': 'oracle',
                'horizon': horizon,
                'checkpoint': str(base_model.__class__.__name__),
                **oracle_metrics,
            },
        ])

    df = pd.DataFrame(comparison_rows).sort_values(['horizon', 'method'])
    csv_path = output_dir / 'horizon_comparison.csv'
    json_path = output_dir / 'horizon_comparison.json'
    plot_path = output_dir / 'horizon_comparison.png'

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    df.to_json(json_path, orient='records', force_ascii=False, indent=4)
    save_comparison_plot(df, plot_path)

    print("\n对比结果摘要:")
    print(df[['method', 'horizon', 'rmse', 'mae', 'acc', 'valid_points']].to_string(index=False))
    print(f"\n对比表已保存: {csv_path}")
    print(f"对比JSON已保存: {json_path}")
    print(f"对比图已保存: {plot_path}")

    return df


def plot_rollout_metrics(per_month_metrics, save_path):
    """绘制滚动预测逐月评估曲线。"""
    if not per_month_metrics:
        return

    months = [m['month_ahead'] for m in per_month_metrics]
    rmse = [m['rmse'] for m in per_month_metrics]
    mae = [m['mae'] for m in per_month_metrics]
    acc = [m['acc'] for m in per_month_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(months, rmse, marker='o', linewidth=2)
    axes[0].set_title('RMSE by Lead Month')
    axes[0].set_xlabel('Month Ahead')
    axes[0].set_ylabel('RMSE (degC)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(months, mae, marker='o', linewidth=2, color='tab:orange')
    axes[1].set_title('MAE by Lead Month')
    axes[1].set_xlabel('Month Ahead')
    axes[1].set_ylabel('MAE (degC)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(months, acc, marker='o', linewidth=2, color='tab:green')
    axes[2].set_title('ACC by Lead Month')
    axes[2].set_xlabel('Month Ahead')
    axes[2].set_ylabel('ACC')
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("开始预测")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    
    # 步骤1: 加载模型
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    print_model_info(model, verbose=False)

    # 如果提供了多个直接多步预测checkpoint，则进入对比模式
    if args.direct_checkpoints:
        print("\n进入 horizon 对比模式...")

        direct_horizons = []
        for ckpt_path in args.direct_checkpoints:
            _, ckpt_args = load_checkpoint_metadata(ckpt_path)
            horizon = int(getattr(ckpt_args, 'output_months', 1))
            direct_horizons.append(horizon)
            print(f"  - 直接模型: {ckpt_path} -> horizon={horizon}")

        max_compare_horizon = max([1] + direct_horizons)
        print(f"  - 对比数据集最大提前期: {max_compare_horizon} 个月")

        print("\n准备对比数据...")
        data_result = create_data_loaders(
            data_path=args.data_path,
            input_months=12,
            output_months=max_compare_horizon,
            batch_size=args.batch_size,
            num_workers=0,
            train_ratio=0.7,
            val_ratio=0.15,
            normalize_method='zscore',
            spatial_downsample=args.spatial_downsample,
            time_range=None
        )

        compare_loader = data_result['test_loader']
        processor = data_result['processor']

        compare_horizon_methods(
            base_model=model,
            direct_checkpoints=args.direct_checkpoints,
            dataloader=compare_loader,
            device=device,
            output_dir=output_dir,
            processor=processor
        )

        print("\n" + "=" * 70)
        print("对比完成！")
        print("=" * 70)
        print(f"输出目录: {output_dir}")
        print("=" * 70)
        return
    
    # 步骤2: 准备数据
    print("\n准备数据...")
    data_result = create_data_loaders(
        data_path=args.data_path,
        input_months=12,  # 应与训练时一致
        output_months=1,
        batch_size=args.batch_size,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.15,
        normalize_method='zscore',
        spatial_downsample=args.spatial_downsample,
        time_range=None
    )
    
    test_loader = data_result['test_loader']
    processor = data_result['processor']

    split_to_dataset = {
        'train': data_result['train_dataset'],
        'val': data_result['val_dataset'],
        'test': data_result['test_dataset']
    }
    
    # 步骤3: 在测试集上进行预测
    print("\n在测试集上进行预测...")
    start_time = time.time()
    
    predictions, targets, masks = predict_on_dataset(model, test_loader, device, processor)
    
    elapsed_time = time.time() - start_time
    print(f"预测完成！耗时: {elapsed_time:.2f}s")
    print(f"预测结果形状: {predictions.shape}")
    
    # 步骤4: 评估
    print("\n评估预测结果...")
    
    # 计算指标（在有效像素上）
    valid_mask = masks.astype(bool)
    
    pred_valid = predictions[valid_mask]
    target_valid = targets[valid_mask]
    
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
    mae = np.mean(np.abs(pred_valid - target_valid))
    
    # 计算ACC
    pred_anom = pred_valid - np.mean(pred_valid)
    target_anom = target_valid - np.mean(target_valid)
    acc = np.sum(pred_anom * target_anom) / (np.sqrt(np.sum(pred_anom**2)) * np.sqrt(np.sum(target_anom**2)) + 1e-8)
    
    print("\n测试结果:")
    print(f"  - RMSE: {rmse:.4f}°C")
    print(f"  - MAE: {mae:.4f}°C")
    print(f"  - ACC: {acc:.4f}")
    
    # 保存评估结果
    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'acc': float(acc),
        'num_samples': int(predictions.shape[0]),
        'checkpoint': str(args.checkpoint)
    }
    
    import json
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"评估结果已保存: {results_path}")
    
    # 步骤5: 保存预测结果
    if args.save_predictions:
        print("\n保存预测结果...")
        predictions_path = output_dir / 'predictions.nc'
        save_predictions_to_netcdf(predictions, str(predictions_path))
        
        targets_path = output_dir / 'targets.nc'
        save_predictions_to_netcdf(targets, str(targets_path))
    
    # 步骤6: 可视化
    if args.visualize and args.num_samples > 0:
        print("\n可视化预测结果...")
        
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # 随机选择一些样本进行可视化
        num_samples = min(args.num_samples, predictions.shape[0])
        indices = np.random.choice(predictions.shape[0], num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            pred_sample = predictions[idx, 0]  # 去除通道维度
            target_sample = targets[idx, 0]
            
            # 获取该样本的掩膜
            mask_sample = valid_mask[idx, 0]
            
            # 只在有效像素上计算指标
            if mask_sample.sum() > 0:
                sample_rmse = np.sqrt(np.mean((pred_sample[mask_sample] - target_sample[mask_sample]) ** 2))
            else:
                sample_rmse = 0.0
            
            # 绘制对比图
            save_path = vis_dir / f'sample_{idx:03d}_rmse{sample_rmse:.2f}.png'
            plot_prediction_comparison(
                target_sample,
                pred_sample,
                title=f'Sample {idx} - RMSE: {sample_rmse:.2f}°C',
                save_path=str(save_path)
            )
        
        print(f"已保存 {num_samples} 个可视化样本到: {vis_dir}")

    # 步骤7: 自回归滚动预测（可选）
    if args.rollout_months > 0:
        print("\n执行自回归滚动预测...")

        rollout_dataset = split_to_dataset[args.rollout_split]
        if len(rollout_dataset) == 0:
            raise ValueError(f"{args.rollout_split} 数据集为空，无法执行滚动预测")

        idx = max(0, min(args.rollout_start_idx, len(rollout_dataset) - 1))
        X0, _, _ = rollout_dataset[idx]

        # dataset返回形状为 (1, T_in, H, W)，补batch维
        init_input = X0.unsqueeze(0)

        rollout_pred = rollout_forecast(
            model=model,
            init_input=init_input,
            months=args.rollout_months,
            device=device
        )

        # 预测结果（归一化尺度）：(months, H, W)
        rollout_pred_norm = rollout_pred.numpy()[0, 0]

        # 获取可用于评估的真实未来序列（同一split下相邻样本对应未来逐月真值）
        max_eval_months = max(0, len(rollout_dataset) - idx)
        eval_months = min(args.rollout_months, max_eval_months)

        if eval_months <= 0:
            print("⚠️ 当前起始索引后无可用真实值，跳过滚动预测精度评估")
            rollout_target_norm = None
            rollout_mask = None
        else:
            rollout_target_norm = rollout_dataset.y[idx:idx + eval_months, 0, 0].cpu().numpy()
            rollout_mask = rollout_dataset.mask[idx:idx + eval_months, 0, 0].cpu().numpy().astype(bool)

        # 反归一化（便于用°C解释指标）
        if hasattr(processor, 'stats') and processor.stats['mean'] is not None:
            mean = processor.stats['mean']
            std = processor.stats['std']
            rollout_pred_np = rollout_pred_norm * std + mean
            if rollout_target_norm is not None:
                rollout_target_np = rollout_target_norm * std + mean
            else:
                rollout_target_np = None
        else:
            rollout_pred_np = rollout_pred_norm
            rollout_target_np = rollout_target_norm

        rollout_path = output_dir / f'rollout_{args.rollout_split}_idx{idx}_{args.rollout_months}m.nc'
        save_predictions_to_netcdf(rollout_pred_np, str(rollout_path))

        # 保存对齐的真实值（仅保存可评估部分）
        if rollout_target_np is not None:
            rollout_target_path = output_dir / f'rollout_target_{args.rollout_split}_idx{idx}_{eval_months}m.nc'
            save_predictions_to_netcdf(rollout_target_np, str(rollout_target_path))

            overall_metrics = compute_metrics_np(
                rollout_pred_np[:eval_months],
                rollout_target_np,
                rollout_mask
            )

            per_month_metrics = []
            for m in range(eval_months):
                m_metrics = compute_metrics_np(
                    rollout_pred_np[m],
                    rollout_target_np[m],
                    rollout_mask[m]
                )
                m_metrics['month_ahead'] = m + 1
                per_month_metrics.append(m_metrics)

            rollout_metrics = {
                'split': args.rollout_split,
                'start_idx': int(idx),
                'requested_months': int(args.rollout_months),
                'evaluated_months': int(eval_months),
                'overall': overall_metrics,
                'per_month': per_month_metrics
            }

            import json
            rollout_metrics_path = output_dir / f'rollout_metrics_{args.rollout_split}_idx{idx}_{eval_months}m.json'
            with open(rollout_metrics_path, 'w', encoding='utf-8') as f:
                json.dump(rollout_metrics, f, indent=4, ensure_ascii=False)

            rollout_plot_path = output_dir / f'rollout_metrics_plot_{args.rollout_split}_idx{idx}_{eval_months}m.png'
            plot_rollout_metrics(per_month_metrics, str(rollout_plot_path))

            print("滚动预测精度评估:")
            print(f"  - 评估月数: {eval_months}")
            print(f"  - Overall RMSE: {overall_metrics['rmse']:.4f}°C")
            print(f"  - Overall MAE: {overall_metrics['mae']:.4f}°C")
            print(f"  - Overall ACC: {overall_metrics['acc']:.4f}")
            print(f"  - 逐月指标文件: {rollout_metrics_path}")
            print(f"  - 指标曲线图: {rollout_plot_path}")

        print("滚动预测完成:")
        print(f"  - 数据集: {args.rollout_split}")
        print(f"  - 起始样本索引: {idx}")
        print(f"  - 预测月数: {args.rollout_months}")
        print(f"  - 结果文件: {rollout_path}")
    
    # 完成
    print("\n" + "=" * 70)
    print("预测完成！")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
