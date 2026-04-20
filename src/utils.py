"""
工具函数模块 - utils.py
=======================

这个模块包含各种实用工具函数：
- 可视化工具（绘制SST图、训练曲线等）
- 评估指标（RMSE、MAE、ACC等）
- 辅助函数（设置随机种子、打印模型信息等）
- 日志工具

新手提示：
- 这些工具函数在训练和评估时都会用到
- 可以根据需要扩展新的功能
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import warnings

# 忽略一些matplotlib的警告
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================
# 随机种子设置
# ============================================

def set_seed(seed: int = 42):
    """
    设置随机种子以保证结果可复现
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置CUDA后端的一些选项（可能影响性能但增加确定性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为: {seed}")


# ============================================
# 模型信息打印
# ============================================

def print_model_info(model: torch.nn.Module, verbose: bool = False):
    """
    打印模型信息
    
    参数:
        model: PyTorch模型
        verbose: 是否打印详细的层信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 70)
    print("模型信息")
    print("=" * 70)
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    
    if verbose:
        print("\n详细层结构:")
        print("-" * 70)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"{name:40s} {str(module.__class__.__name__):20s} {num_params:10,} params")
        print("-" * 70)
    
    print("=" * 70 + "\n")


# ============================================
# 评估指标
# ============================================

def calculate_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    计算RMSE（均方根误差）
    
    参数:
        pred: 预测值
        target: 真实值
        mask: 掩膜（可选）
    
    返回:
        rmse: RMSE值
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    return rmse.item()


def calculate_mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    计算MAE（平均绝对误差）
    
    参数:
        pred: 预测值
        target: 真实值
        mask: 掩膜（可选）
    
    返回:
        mae: MAE值
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    mae = F.l1_loss(pred, target)
    return mae.item()


def calculate_acc(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    计算ACC（Anomaly Correlation Coefficient，异常相关系数）
    
    这是气象预测中常用的评估指标
    
    参数:
        pred: 预测值
        target: 真实值
        mask: 掩膜（可选）
    
    返回:
        acc: ACC值，范围[-1, 1]，越接近1越好
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    # 去均值（异常值）
    pred_anom = pred - pred.mean()
    target_anom = target - target.mean()
    
    # 计算相关系数
    numerator = (pred_anom * target_anom).sum()
    denominator = torch.sqrt((pred_anom ** 2).sum() * (target_anom ** 2).sum())
    
    if denominator == 0:
        return 0.0
    
    acc = numerator / denominator
    return acc.item()


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                   device: torch.device, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    评估模型在数据集上的性能
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        mask: 全局掩膜（可选）
    
    返回:
        metrics: 包含各种评估指标的字典
    """
    model.eval()
    
    total_rmse = 0.0
    total_mae = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X, y, batch_mask in dataloader:
            X = X.to(device)
            y = y.to(device)
            batch_mask = batch_mask.to(device)
            
            # 前向传播
            pred = model(X)
            
            # 计算指标
            total_rmse += calculate_rmse(pred, y, batch_mask)
            total_mae += calculate_mae(pred, y, batch_mask)
            total_acc += calculate_acc(pred, y, batch_mask)
            num_batches += 1
    
    # 计算平均
    metrics = {
        'RMSE': total_rmse / num_batches,
        'MAE': total_mae / num_batches,
        'ACC': total_acc / num_batches
    }
    
    return metrics


# ==========================================
# 可视化工具
# ==========================================

def visualize_sst_map(sst_data: np.ndarray, title: str = "SST Map", 
                     save_path: Optional[str] = None, cmap: str = 'jet',
                     vmin: float = -2, vmax: float = 35):
    """
    可视化SST空间分布图
    
    参数:
        sst_data: SST数据，形状 (H, W) 或 (1, H, W)
        title: 图标题
        save_path: 保存路径（可选）
        cmap: 颜色映射
        vmin, vmax: 颜色范围
    """
    # 处理输入形状
    if sst_data.ndim == 3:
        sst_data = sst_data[0]  # 去除通道维度
    
    plt.figure(figsize=(12, 5))
    plt.imshow(sst_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(label='SST (°C)')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_sst_timeseries(time_series: np.ndarray, dates: Optional[np.ndarray] = None,
                            title: str = "SST Time Series", save_path: Optional[str] = None):
    """
    可视化SST时间序列
    
    参数:
        time_series: 时间序列数据，形状 (T,)
        dates: 日期数组（可选）
        title: 图标题
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 4))
    
    if dates is not None:
        plt.plot(dates, time_series, linewidth=1.5)
        plt.xticks(rotation=45)
    else:
        plt.plot(time_series, linewidth=1.5)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('SST (°C)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        save_path: Optional[str] = None):
    """
    绘制训练曲线
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_comparison(y_true: np.ndarray, y_pred: np.ndarray,
                               title: str = "Prediction vs Ground Truth",
                               save_path: Optional[str] = None):
    """
    绘制预测值与真实值的对比图
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        title: 图标题
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 真实值
    im0 = axes[0].imshow(y_true, cmap='jet', vmin=-2, vmax=35)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im0, ax=axes[0], label='SST (°C)')
    
    # 预测值
    im1 = axes[1].imshow(y_pred, cmap='jet', vmin=-2, vmax=35)
    axes[1].set_title('Prediction')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[1], label='SST (°C)')
    
    # 误差
    error = np.abs(y_pred - y_true)
    im2 = axes[2].imshow(error, cmap='Reds', vmin=0)
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[2], label='Error (°C)')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ==========================================
# 辅助工具函数
# ==========================================

def get_device():
    """
    获取可用的计算设备
    
    返回:
        device: torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    参数:
        model: PyTorch模型
    
    返回:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    """
    将秒数格式化为易读的时间字符串
    
    参数:
        seconds: 秒数
    
    返回:
        格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def save_model_info(model: torch.nn.Module, save_path: str, config: Dict = None):
    """
    保存模型信息到文件
    
    参数:
        model: PyTorch模型
        save_path: 保存路径
        config: 配置字典（可选）
    """
    total_params, trainable_params = count_parameters(model)
    
    info = {
        'model_architecture': str(model),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # 假设float32，4字节
    }
    
    if config is not None:
        info['config'] = config
    
    # 保存为JSON
    import json
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"模型信息已保存: {save_path}")


# ==========================================
# 测试代码
# ==========================================

if __name__ == '__main__':
    print("=" * 70)
    print("测试 utils.py 中的工具函数")
    print("=" * 70)
    
    # 测试1: 设置随机种子
    print("\n1. 测试 set_seed()")
    set_seed(42)
    print("随机种子设置完成")
    
    # 测试2: 获取设备
    print("\n2. 测试 get_device()")
    device = get_device()
    print(f"设备: {device}")
    
    # 测试3: 格式化时间
    print("\n3. 测试 format_time()")
    test_times = [45, 120, 3665, 7200]
    for t in test_times:
        print(f"  {t}秒 -> {format_time(t)}")
    
    # 测试4: 可视化函数（仅在可以显示图形时测试）
    print("\n4. 测试可视化函数")
    try:
        # 生成测试数据
        test_sst = np.random.randn(45, 90) * 5 + 15  # 模拟SST数据
        
        # 测试时间序列可视化
        time_series = np.random.randn(120).cumsum() + 15
        visualize_sst_timeseries(time_series, title="Test Time Series")
        
        # 测试空间分布可视化
        visualize_sst_map(test_sst, title="Test SST Map")
        
        # 测试预测对比可视化
        y_true = test_sst
        y_pred = test_sst + np.random.randn(*test_sst.shape) * 0.5
        plot_prediction_comparison(y_true, y_pred, title="Test Prediction")
        
        print("可视化函数测试完成！")
        
    except Exception as e:
        print(f"可视化测试跳过: {e}")
        print("（可能是因为没有GUI环境）")
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
