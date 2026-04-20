# ============================================
# 3D Swin Transformer for SST Prediction
# 海表温度预测的3D Swin Transformer模型
# ============================================

## 📁 项目结构

```
D:\QinBo\Sea-Interface-Temperature-Prodictor\
├── data/                          # 数据目录（需要你放入HadISST_sst.nc）
│   └── HadISST_sst.nc            # 原始NetCDF数据文件
├── src/                           # 源代码目录
│   ├── __init__.py               # 包初始化
│   ├── data_loader.py            # 数据加载器（新手第一课）
│   ├── model_3dswin.py           # 3D Swin Transformer模型（核心）
│   ├── train.py                  # 训练脚本
│   ├── predict.py                # 预测脚本
│   └── utils.py                  # 工具函数
├── configs/                       # 配置文件
│   └── config.yaml               # 默认配置
├── checkpoints/                   # 模型检查点保存目录
├── outputs/                       # 预测输出目录
├── notebooks/                     # Jupyter笔记本（可选）
│   └── tutorial.ipynb            # 交互式教程
└── README.md                      # 本文件
```

## 🚀 快速开始

### 第1步：环境准备
```bash
# 1. 确保你在项目根目录
cd D:\QinBo\Sea-Interface-Temperature-Prodictor

# 2. 激活虚拟环境（Windows）
.venv\Scripts\activate

# 3. 安装依赖包（第一次需要）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xarray netCDF4 numpy matplotlib pyyaml einops pytorch-lightning
```

### 第2步：数据准备
确保 `data/HadISST_sst.nc` 文件存在（从项目README提供的链接下载）

### 第3步：运行数据探索（第一课）
```bash
# 查看数据基本信息
python -c "from src.data_loader import SSTDataLoader; loader = SSTDataLoader(); print(loader)"
```

### 第4步：开始训练（当你理解代码后）
```bash
# 使用默认配置训练
python src/train.py --config configs/config.yaml

# 或者覆盖某些参数
python src/train.py --config configs/config.yaml --batch_size 2 --epochs 10
```

### 第5步：进行预测
```bash
# 使用训练好的模型预测
python src/predict.py --checkpoint checkpoints/best_model.ckpt --output outputs/prediction.nc
```

