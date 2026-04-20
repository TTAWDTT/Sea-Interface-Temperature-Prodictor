"""
SST Prediction with 3D Swin Transformer
=========================================

海表温度预测的3D Swin Transformer实现

模块说明:
- data_loader: 数据加载和预处理
- model_3dswin: 3D Swin Transformer模型
- train: 训练脚本
- predict: 预测/推理脚本
- utils: 工具函数

作者: Sisyphus
日期: 2024
"""

__version__ = "0.1.0"
__author__ = "Sisyphus"

# 方便导入的主要类和函数
from .data_loader import (
    SSTDataProcessor,
    SSTDataset,
    create_data_loaders,
    denormalize_predictions,
    save_predictions
)

from .model_3dswin import (
    SwinTransformer3D,
    build_swin_3d_tiny,
    build_swin_3d_small,
    build_swin_3d_base,
    WindowAttention3D,
    SwinTransformer3DBlock,
    PatchEmbed3D,
    PatchMerging3D
)

# 版本信息
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'description': '3D Swin Transformer for SST Prediction',
    'python_requires': '>=3.8',
    'dependencies': [
        'torch>=1.12.0',
        'numpy>=1.21.0',
        'xarray>=0.20.0',
        'netCDF4>=1.5.8',
        'matplotlib>=3.4.0',
        'pyyaml>=5.4.0',
        'einops>=0.4.0',
        'pytorch-lightning>=1.6.0',
        'tensorboard>=2.8.0'
    ]
}


def print_version_info():
    """打印版本信息"""
    print(f"SST-3D-Swin-Transformer v{VERSION_INFO['version']}")
    print(f"Author: {VERSION_INFO['author']}")
    print(f"Description: {VERSION_INFO['description']}")


def check_dependencies():
    """检查依赖是否安装"""
    import importlib
    
    missing = []
    for dep in VERSION_INFO['dependencies']:
        package = dep.split('>=')[0]
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            missing.append(dep)
    
    if missing:
        print("警告：以下依赖未安装:")
        for dep in missing:
            print(f"  - {dep}")
        print("\n请使用以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("所有依赖已安装！")
        return True


if __name__ == '__main__':
    # 当直接运行这个文件时，打印版本信息
    print_version_info()
    check_dependencies()
