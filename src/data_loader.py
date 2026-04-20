"""
数据加载器模块 - SSTDataLoader
===============================
这个模块负责：
1. 读取NetCDF格式的SST数据
2. 数据预处理和归一化
3. 创建PyTorch Dataset和DataLoader
4. 处理缺失值和掩膜

新手提示：
- NetCDF是一种科学数据格式，常用于存储气象/海洋数据
- xarray是处理NetCDF的Python库
- Dataset是PyTorch的数据抽象，DataLoader负责批量加载
"""

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings


class SSTDataProcessor:
    """
    SST数据预处理器
    
    负责：
    - 读取原始NetCDF数据
    - 处理缺失值（哨兵值）
    - 数据归一化/标准化
    - 创建时间序列样本
    
    参数:
        data_path (str): NetCDF文件路径
        input_months (int): 输入序列长度（过去多少个月）
        output_months (int): 输出序列长度（预测多少个月）
        normalize_method (str): 归一化方法，可选'zscore', 'minmax', 'none'
    """
    
    def __init__(
        self,
        data_path: str = "HadISST_sst.nc",
        input_months: int = 12,
        output_months: int = 1,
        normalize_method: str = "zscore",
        sentinel_threshold: float = -100.0
    ):
        self.data_path = Path(data_path)
        self.input_months = input_months
        self.output_months = output_months
        self.normalize_method = normalize_method
        self.sentinel_threshold = sentinel_threshold
        
        # 数据统计（用于归一化）
        self.stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None
        }
        
        # 原始数据缓存
        self._raw_data = None
        self._masked_data = None
        self._dates = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载并预处理原始数据
        
        返回:
            data (np.ndarray): 处理后的SST数据，形状为(T, H, W)
            dates (np.ndarray): 对应的时间数组
        """
        print(f"正在加载数据: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"找不到数据文件: {self.data_path}\n"
                "请从 https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html "
                "下载 HadISST_sst.nc.gz 并解压到项目根目录"
            )
        
        # 使用xarray打开NetCDF
        ds = xr.open_dataset(self.data_path, decode_times=True)
        
        print(f"数据集信息:")
        print(f"  - 时间范围: {ds.time.values[0]} 到 {ds.time.values[-1]}")
        print(f"  - 空间维度: {len(ds.latitude)} lat × {len(ds.longitude)} lon")
        print(f"  - 总时间步: {len(ds.time)}")
        
        # 提取SST数据
        sst = ds['sst']

        # 直接加载为numpy，再原地处理哨兵值，避免xarray.where带来的额外峰值内存
        print(f"\n掩膜哨兵值 (< {self.sentinel_threshold})...")
        data = np.asarray(sst.values, dtype=np.float32).copy()  # 形状: (T, lat, lon)
        data[data <= self.sentinel_threshold] = np.nan

        dates = ds.time.values
        
        # 保存统计信息
        valid_mask = ~np.isnan(data)
        if valid_mask.any():
            self.stats['mean'] = float(data[valid_mask].mean())
            self.stats['std'] = float(data[valid_mask].std())
            self.stats['min'] = float(data[valid_mask].min())
            self.stats['max'] = float(data[valid_mask].max())
        
        print(f"\n数据统计:")
        print(f"  - 有效数据比例: {valid_mask.sum() / valid_mask.size:.2%}")
        print(f"  - 均值: {self.stats['mean']:.2f}°C")
        print(f"  - 标准差: {self.stats['std']:.2f}°C")
        print(f"  - 范围: [{self.stats['min']:.2f}, {self.stats['max']:.2f}]°C")
        
        # 缓存
        self._raw_data = data.copy()
        self._masked_data = data.copy()
        self._dates = dates.copy()
        
        ds.close()
        
        return data, dates
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        归一化数据
        
        参数:
            data (np.ndarray): 原始数据
            
        返回:
            normalized_data (np.ndarray): 归一化后的数据
        """
        if self.normalize_method == 'zscore':
            # Z-score标准化: (x - mean) / std
            return (data - self.stats['mean']) / (self.stats['std'] + 1e-8)
        
        elif self.normalize_method == 'minmax':
            # Min-Max归一化: (x - min) / (max - min)
            return (data - self.stats['min']) / (self.stats['max'] - self.stats['min'] + 1e-8)
        
        elif self.normalize_method == 'none':
            return data
        
        else:
            raise ValueError(f"未知的归一化方法: {self.normalize_method}")
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        反归一化（将归一化后的数据还原为原始尺度）
        
        参数:
            data (np.ndarray): 归一化后的数据
            
        返回:
            original_data (np.ndarray): 还原后的数据
        """
        if self.normalize_method == 'zscore':
            return data * self.stats['std'] + self.stats['mean']
        
        elif self.normalize_method == 'minmax':
            return data * (self.stats['max'] - self.stats['min']) + self.stats['min']
        
        elif self.normalize_method == 'none':
            return data
        
        else:
            raise ValueError(f"未知的归一化方法: {self.normalize_method}")
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建输入-输出序列对
        
        例如：input_months=12, output_months=1
        输入：过去12个月的SST场 → 输出：下1个月的SST场
        
        参数:
            data (np.ndarray): 完整的时间序列数据 (T, H, W)
            
        返回:
            X (np.ndarray): 输入序列 (N, input_months, H, W)
            y (np.ndarray): 输出序列 (N, output_months, H, W)
        """
        T, H, W = data.shape
        sequences = []
        
        # 计算可以创建多少个样本
        # 需要：input_months + output_months 个连续时间步
        num_samples = T - self.input_months - self.output_months + 1
        
        if num_samples <= 0:
            raise ValueError(
                f"时间序列太短！需要至少 {self.input_months + self.output_months} 个月，"
                f"但只有 {T} 个月"
            )
        
        print(f"\n创建序列:")
        print(f"  - 总时间步: {T}")
        print(f"  - 输入长度: {self.input_months}")
        print(f"  - 输出长度: {self.output_months}")
        print(f"  - 可创建样本数: {num_samples}")
        
        X_list = []
        y_list = []
        
        for i in range(num_samples):
            # 输入：从i开始，长度为input_months
            X_seq = data[i:i+self.input_months]  # (input_months, H, W)
            
            # 输出：紧接着的output_months个月
            y_seq = data[i+self.input_months:i+self.input_months+self.output_months]  # (output_months, H, W)
            
            X_list.append(X_seq)
            y_list.append(y_seq)
        
        X = np.array(X_list)  # (num_samples, input_months, H, W)
        y = np.array(y_list)  # (num_samples, output_months, H, W)
        
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        
        return X, y


class SSTDataset(Dataset):
    """
    PyTorch Dataset for SST数据
    
    这是PyTorch的数据集类，用于批量加载数据
    
    参数:
        X (np.ndarray): 输入数据 (N, T_in, H, W)
        y (np.ndarray): 输出数据 (N, T_out, H, W)
        mask (np.ndarray, optional): 掩膜数据（标识有效像素）
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None):
        super().__init__()

        # 先基于原始y构建有效掩膜，再清洗NaN，避免NaN在卷积中扩散导致loss=nan
        if mask is not None:
            self.mask = torch.from_numpy(mask).bool()
        else:
            y_tensor_raw = torch.from_numpy(y).float().unsqueeze(1)
            self.mask = torch.isfinite(y_tensor_raw) & (y_tensor_raw > -100)

        # 清洗输入与目标中的NaN/Inf，模型前向与loss只在mask有效区域计算
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # 转换为torch张量
        # 注意：我们添加通道维度，变成 (N, C, T, H, W)
        # 对于单变量SST，C=1
        self.X = torch.from_numpy(X_clean).float().unsqueeze(1)  # (N, 1, T_in, H, W)
        self.y = torch.from_numpy(y_clean).float().unsqueeze(1)  # (N, 1, T_out, H, W)
        
        print(f"Dataset created:")
        print(f"  - X shape: {self.X.shape}")
        print(f"  - y shape: {self.y.shape}")
        print(f"  - mask shape: {self.mask.shape}")
        print(f"  - valid pixels: {self.mask.float().mean():.2%}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        返回:
            X: 输入数据 (1, T_in, H, W)
            y: 输出数据 (1, T_out, H, W)
            mask: 掩膜 (1, T_out, H, W)
        """
        return self.X[idx], self.y[idx], self.mask[idx]


def create_data_loaders(
    data_path: str = "HadISST_sst.nc",
    input_months: int = 12,
    output_months: int = 1,
    batch_size: int = 2,
    num_workers: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    normalize_method: str = "zscore",
    spatial_downsample: int = 1,  # 空间降采样因子（1=不降采样，2=减半）
    time_range: Optional[Tuple[str, str]] = None,  # 时间范围筛选，如 ('1980-01', '2020-12')
) -> Dict:
    """
    创建训练、验证、测试的数据加载器
    
    这是主要的接口函数，新手从这里开始理解数据流！
    
    参数:
        data_path: 数据文件路径
        input_months: 输入时间步长（月）
        output_months: 输出时间步长（月）
        batch_size: 批量大小
        num_workers: 数据加载线程数（Windows建议0）
        train_ratio: 训练集比例
        val_ratio: 验证集比例（剩余的是测试集）
        normalize_method: 归一化方法
        spatial_downsample: 空间降采样因子
        time_range: 筛选特定时间范围，如('1980-01', '2020-12')
    
    返回:
        dict: 包含以下键的字典
            - 'train_loader': 训练集DataLoader
            - 'val_loader': 验证集DataLoader
            - 'test_loader': 测试集DataLoader
            - 'processor': SSTDataProcessor实例（用于反归一化）
            - 'stats': 数据统计信息
    """
    
    print("=" * 60)
    print("创建数据加载器")
    print("=" * 60)
    
    # 步骤1: 创建数据处理器并加载数据
    processor = SSTDataProcessor(
        data_path=data_path,
        input_months=input_months,
        output_months=output_months,
        normalize_method=normalize_method
    )
    
    # 加载原始数据
    raw_data, dates = processor.load_data()
    
    # 步骤2: 时间范围筛选（如果指定）
    if time_range is not None:
        start_time, end_time = time_range
        start_idx = np.searchsorted(dates, np.datetime64(start_time))
        end_idx = np.searchsorted(dates, np.datetime64(end_time))
        
        raw_data = raw_data[start_idx:end_idx+1]
        dates = dates[start_idx:end_idx+1]
        
        print(f"\n时间筛选: {start_time} 到 {end_time}")
        print(f"  - 筛选后时间步: {len(dates)}")
    
    # 步骤3: 空间降采样（如果指定）
    if spatial_downsample > 1:
        print(f"\n空间降采样: 因子={spatial_downsample}")
        print(f"  - 原始尺寸: {raw_data.shape[1]} x {raw_data.shape[2]}")

        T, H, W = raw_data.shape
        new_H = H // spatial_downsample
        new_W = W // spatial_downsample

        # 使用block pooling并忽略NaN，避免陆地/缺测区域通过卷积扩散到海洋像素
        cropped = raw_data[:, :new_H * spatial_downsample, :new_W * spatial_downsample]
        pooled = cropped.reshape(
            T,
            new_H,
            spatial_downsample,
            new_W,
            spatial_downsample
        )
        valid = np.isfinite(pooled)
        pooled_sum = np.where(valid, pooled, 0.0).sum(axis=(2, 4), dtype=np.float32)
        pooled_count = valid.sum(axis=(2, 4))
        raw_data = np.divide(
            pooled_sum,
            pooled_count,
            out=np.full((T, new_H, new_W), np.nan, dtype=np.float32),
            where=pooled_count > 0
        )

        print(f"  - 降采样后尺寸: {raw_data.shape[1]} x {raw_data.shape[2]}")
    
    # 步骤4: 归一化
    print(f"\n归一化: 方法={normalize_method}")
    data_normalized = processor.normalize(raw_data)
    
    # 步骤5: 创建序列
    print("\n创建时间序列样本...")
    X, y = processor.create_sequences(data_normalized)
    
    # 步骤6: 划分训练/验证/测试集（按时间顺序！）
    print("\n划分数据集（按时间顺序）...")
    
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # 测试集是剩余部分
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    print(f"  - 训练集: {len(X_train)} 样本 ({len(X_train)/n_total:.1%})")
    print(f"  - 验证集: {len(X_val)} 样本 ({len(X_val)/n_total:.1%})")
    print(f"  - 测试集: {len(X_test)} 样本 ({len(X_test)/n_total:.1%})")
    
    # 步骤7: 创建PyTorch Datasets
    print("\n创建PyTorch DataLoaders...")
    
    # 可选：为训练集添加掩膜（用于处理陆地/缺失区域）
    # 这里我们使用简单的NaN掩膜
    train_dataset = SSTDataset(X_train, y_train)
    val_dataset = SSTDataset(X_val, y_val)
    test_dataset = SSTDataset(X_test, y_test)
    
    # 创建DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # 丢弃不完整的最后一个batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时不打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("\n" + "=" * 60)
    print("数据加载器创建完成!")
    print("=" * 60)
    print(f"批次大小: {batch_size}")
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    print(f"测试批次: {len(test_loader)}")
    print("=" * 60)
    
    # 返回所有需要的信息
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'processor': processor,
        'stats': processor.stats,
        'raw_data_shape': raw_data.shape,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


def denormalize_predictions(predictions: np.ndarray, processor: SSTDataProcessor) -> np.ndarray:
    """
    将模型预测的反归一化回原始温度尺度
    
    参数:
        predictions (np.ndarray): 模型输出的归一化预测
        processor (SSTDataProcessor): 数据处理器（包含统计信息）
        
    返回:
        original_scale (np.ndarray): 反归一化后的预测
    """
    return processor.denormalize(predictions)


def save_predictions(
    predictions: np.ndarray,
    save_path: str,
    dates: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None
):
    """
    保存预测结果为NetCDF格式
    
    参数:
        predictions (np.ndarray): 预测结果 (T, H, W) 或 (T, 1, H, W)
        save_path (str): 保存路径
        dates (np.ndarray, optional): 时间坐标
        lat (np.ndarray, optional): 纬度坐标
        lon (np.ndarray, optional): 经度坐标
    """
    # 确保 predictions 是 (T, H, W)
    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    
    # 创建DataArray
    if dates is None:
        dates = np.arange(len(predictions))
    
    if lat is None:
        lat = np.arange(predictions.shape[1])
    
    if lon is None:
        lon = np.arange(predictions.shape[2])
    
    da = xr.DataArray(
        predictions,
        coords=[dates, lat, lon],
        dims=['time', 'latitude', 'longitude'],
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


# ============================================
# 测试代码（当直接运行此文件时）
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 SSTDataLoader")
    print("=" * 60)
    
    # 测试配置
    config = {
        'data_path': 'HadISST_sst.nc',
        'input_months': 12,
        'output_months': 1,
        'batch_size': 2,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'normalize_method': 'zscore',
        'spatial_downsample': 4,  # 测试时降采样4倍以节省内存
        'time_range': ('1980-01', '2020-12'),  # 只用最近40年数据
    }
    
    try:
        # 创建数据加载器
        result = create_data_loaders(**config)
        
        # 测试加载一个batch
        train_loader = result['train_loader']
        
        print("\n" + "=" * 60)
        print("测试加载一个batch")
        print("=" * 60)
        
        for batch_idx, (X, y, mask) in enumerate(train_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  X shape: {X.shape} (batch, channel, time, height, width)")
            print(f"  y shape: {y.shape}")
            print(f"  mask shape: {mask.shape}")
            print(f"  X value range: [{X.min():.2f}, {X.max():.2f}]")
            print(f"  y value range: [{y.min():.2f}, {y.max():.2f}]")
            
            if batch_idx == 0:
                break  # 只测试第一个batch
        
        print("\n" + "=" * 60)
        print("数据加载器测试成功！")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请确保数据文件存在:")
        print("  1. 从 https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html 下载")
        print("  2. 解压 HadISST_sst.nc.gz")
        print("  3. 将 HadISST_sst.nc 放在项目根目录")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
