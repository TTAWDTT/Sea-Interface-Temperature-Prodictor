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
import torch
import xarray as xr

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    # 确保 predictions 是 (N, T, H, W) 或 (N, H, W)
    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]  # 去除通道维度
    
    N, *spatial_dims = predictions.shape
    
    # 创建坐标
    if dates is None:
        dates = np.arange(N)
    
    if len(spatial_dims) == 3:  # (T, H, W)
        T, H, W = spatial_dims
        if lat is None:
            lat = np.arange(H)
        if lon is None:
            lon = np.arange(W)
        
        coords = [dates, np.arange(T), lat, lon]
        dims = ['sample', 'time', 'latitude', 'longitude']
        
    else:  # (H, W)
        H, W = spatial_dims
        if lat is None:
            lat = np.arange(H)
        if lon is None:
            lon = np.arange(W)
        
        coords = [dates, lat, lon]
        dims = ['sample', 'latitude', 'longitude']
    
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
    
    # 完成
    print("\n" + "=" * 70)
    print("预测完成！")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
