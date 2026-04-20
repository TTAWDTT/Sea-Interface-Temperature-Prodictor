# 3D Swin Transformer for SST Prediction

## 项目概述

本项目使用 **3D Swin Transformer** 深度学习模型来预测海表温度（Sea Surface Temperature, SST）。

### 什么是3D Swin Transformer？

3D Swin Transformer 是一种先进的深度学习架构，它：
- **处理时空数据**：可以同时处理时间维度（月份）和空间维度（经纬度）
- **分层特征提取**：从局部到全局逐步提取特征
- **高效注意力机制**：使用窗口注意力减少计算量

### 项目特点

✅ **完整的数据处理流程**：从原始NetCDF数据到模型输入
✅ **灵活的模型配置**：支持Tiny/Small/Base等多种模型规模
✅ **高效的训练系统**：支持混合精度、梯度裁剪、学习率调度
✅ **完善的评估工具**：RMSE、MAE、ACC等多种指标
✅ **可视化支持**：训练曲线、预测对比图、空间分布图
✅ **详细的文档**：每行代码都有详细注释

## 项目结构

```
D:\QinBo\Sea-Interface-Temperature-Prodictor\
├── configs/
│   └── config.yaml              # 默认配置文件
├── src/
│   ├── __init__.py              # 包初始化
│   ├── data_loader.py           # 数据加载和预处理 ⭐新手从这里开始
│   ├── model_3dswin.py          # 3D Swin Transformer模型 ⭐核心代码
│   ├── train.py                 # 训练脚本
│   ├── predict.py               # 预测脚本
│   └── utils.py                 # 工具函数
├── notebooks/
│   └── tutorial.ipynb           # 交互式教程（可选）
├── data/                        # 数据目录（需要自己下载）
│   └── HadISST_sst.nc          # SST数据文件
├── outputs/                     # 输出目录
│   └── {exp_name}/             # 实验结果
│       ├── checkpoints/        # 模型检查点
│       ├── logs/               # TensorBoard日志
│       └── visualizations/     # 可视化结果
├── README.md                    # 本文件 ⭐请先阅读！
├── README_SwinTransformer.md  # 详细的技术文档
└── requirements.txt             # Python依赖包

```

## 快速开始

### 1. 环境准备

#### 1.1 安装Python依赖

```bash
# 进入项目目录
cd D:\QinBo\Sea-Interface-Temperature-Prodictor

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xarray netCDF4 numpy matplotlib pyyaml einops tensorboard
```

#### 1.2 下载SST数据

1. 访问 [HadISST官方下载页面](https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html)
2. 下载 `HadISST_sst.nc.gz` 文件
3. 解压并将 `HadISST_sst.nc` 放到项目根目录的 `data/` 文件夹中

### 2. 测试数据加载（新手第一步）

```bash
# 测试数据加载器是否正常工作
python -c "from src.data_loader import create_data_loaders; result = create_data_loaders(spatial_downsample=4, batch_size=2); print('✅ 数据加载成功！')"
```

### 3. 开始训练

#### 方式一：使用默认配置

```bash
python src/train.py
```

#### 方式二：使用配置文件

```bash
# 编辑 configs/config.yaml 文件
# 然后运行
python src/train.py --config configs/config.yaml
```

#### 方式三：命令行参数覆盖

```bash
python src/train.py \
    --batch_size 4 \
    --epochs 100 \
    --lr 5e-5 \
    --model_type small \
    --spatial_downsample 2
```

### 4. 使用训练好的模型进行预测

```bash
python src/predict.py \
    --checkpoint outputs/{exp_name}/checkpoints/model_best.pth \
    --output_dir outputs/predictions \
    --visualize \
    --save_predictions \
    --num_samples 10
```

### 5. 使用TensorBoard查看训练过程

```bash
# 启动TensorBoard
tensorboard --logdir outputs

# 然后在浏览器中打开 http://localhost:6006
```