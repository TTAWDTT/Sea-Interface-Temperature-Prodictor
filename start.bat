@echo off
chcp 65001 >nul
REM =========================================================
REM SST预测项目启动脚本
REM 3D Swin Transformer for SST Prediction
REM =========================================================

echo.
echo =========================================================
echo    3D Swin Transformer - SST预测系统
echo =========================================================
echo.

REM 检查虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo [1/4] 正在激活虚拟环境...
    call .venv\Scripts\activate.bat
    echo √ 虚拟环境已激活
) else (
    echo [!] 警告: 未找到虚拟环境，使用系统Python
)

echo.
echo [2/4] 检查必要文件...

REM 检查关键文件
if not exist "src\data_loader.py" (
    echo [X] 错误: 找不到 src\data_loader.py
    echo     请确保在项目根目录运行此脚本
    pause
    exit /b 1
)

if not exist "src\model_3dswin.py" (
    echo [X] 错误: 找不到 src\model_3dswin.py
    pause
    exit /b 1
)

echo √ 所有必要文件已找到

echo.
echo [3/4] 检查数据文件...

REM 检查数据文件
if not exist "HadISST_sst.nc" (
    if not exist "data\HadISST_sst.nc" (
        echo [!] 警告: 未找到 HadISST_sst.nc 数据文件
        echo.
        echo     请按以下步骤获取数据：
        echo     1. 访问 https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html
        echo     2. 下载 HadISST_sst.nc.gz
        echo     3. 解压并将 HadISST_sst.nc 放到项目根目录
        echo.
        echo     或者运行快速测试（不需要真实数据）：
        echo     python -c "print('测试成功！')"
        echo.
        choice /C YN /M "是否继续运行"
        if errorlevel 2 exit /b 0
    ) else (
        echo √ 找到数据文件: data\HadISST_sst.nc
    )
) else (
    echo √ 找到数据文件: HadISST_sst.nc
)

echo.
echo [4/4] 选择操作模式...
echo.

:menu
echo =========================================================
echo                    操作菜单
echo =========================================================
echo.
echo   [1] 快速测试 - 测试数据加载和模型
echo   [2] 快速训练 - 运行1轮训练（测试用）
echo   [3] 完整训练 - 运行完整训练（50轮）
echo   [4] 使用配置文件训练
echo   [5] TensorBoard 可视化
echo   [6] 预测（需要训练好的模型）
echo   [7] 查看帮助文档
echo.
echo   [0] 退出
echo =========================================================
echo.

set /p choice="请输入选项 [0-7]: "

if "%choice%"=="0" goto :exit
if "%choice%"=="1" goto :test
if "%choice%"=="2" goto :quick_train
if "%choice%"=="3" goto :full_train
if "%choice%"=="4" goto :config_train
if "%choice%"=="5" goto :tensorboard
if "%choice%"=="6" goto :predict
if "%choice%"=="7" goto :help

echo [!] 无效选项，请重新输入
goto :menu

:test
echo.
echo =========================================================
echo                   快速测试模式
echo =========================================================
echo.
echo [测试1/3] 测试数据加载...
python -c "from src.data_loader import create_data_loaders; result = create_data_loaders(spatial_downsample=4, batch_size=2); print('\n✓ 数据加载测试成功！')"
if errorlevel 1 goto :test_failed

echo.
echo [测试2/3] 测试模型构建...
python -c "from src.model_3dswin import build_swin_3d_tiny; model = build_swin_3d_tiny(); print('\n✓ 模型构建测试成功！')"
if errorlevel 1 goto :test_failed

echo.
echo [测试3/3] 测试模型前向传播...
python -c "
import torch
from src.model_3dswin import build_swin_3d_tiny
model = build_swin_3d_tiny()
x = torch.randn(1, 1, 12, 56, 56)
with torch.no_grad():
    y = model(x)
print(f'\n✓ 前向传播测试成功！')
print(f'  输入: {x.shape}')
print(f'  输出: {y.shape}')
"
if errorlevel 1 goto :test_failed

echo.
echo =========================================================
echo                   所有测试通过！✓
echo =========================================================
echo.
pause
goto :menu

:test_failed
echo.
echo [X] 测试失败！请检查错误信息。
echo.
pause
goto :menu

:quick_train
echo.
echo =========================================================
echo                   快速训练模式
echo =========================================================
echo.
echo 配置:
echo   - 轮数: 1
echo   - 降采样: 4倍
echo   - 批次: 2
echo   - 模型: Tiny
echo.
echo 开始训练...
echo.
python src/train.py --epochs 1 --spatial_downsample 4 --batch_size 2 --model_type tiny --log_freq 1
echo.
pause
goto :menu

:full_train
echo.
echo =========================================================
echo                   完整训练模式
echo =========================================================
echo.
echo 配置:
echo   - 轮数: 50
echo   - 降采样: 4倍
echo   - 批次: 2
echo   - 模型: Tiny
echo   - 预计时间: 约4-6小时（取决于硬件）
echo.
set /p confirm="确认开始训练吗？(Y/N): "
if /i not "%confirm%"=="Y" goto :menu

echo.
echo 开始训练...
echo.
python src/train.py --epochs 50 --spatial_downsample 4 --batch_size 2 --model_type tiny
echo.
pause
goto :menu

:config_train
echo.
echo =========================================================
echo                   配置文件训练模式
echo =========================================================
echo.
echo 可用配置文件:
dir /b configs\*.yaml 2>nul
if errorlevel 1 (
    echo [警告] 未找到配置文件
    echo.
    echo 请创建配置文件（如 configs/my_config.yaml）
    pause
    goto :menu
)
echo.
set /p config_file="请输入配置文件名（如 config.yaml）: "
if not exist "configs\%config_file%" (
    echo [错误] 配置文件不存在: configs\%config_file%
    pause
    goto :menu
)

echo.
echo 使用配置文件: configs\%config_file%
echo.
python src/train.py --config configs\%config_file%
echo.
pause
goto :menu

:tensorboard
echo.
echo =========================================================
echo                   TensorBoard 可视化
echo =========================================================
echo.
echo 启动 TensorBoard...
echo 请在浏览器中访问: http://localhost:6006
echo.
echo 按 Ctrl+C 停止 TensorBoard
echo.
tensorboard --logdir outputs
echo.
pause
goto :menu

:predict
echo.
echo =========================================================
echo                   预测模式
echo =========================================================
echo.
echo 可用检查点:
dir /s /b outputs\*\checkpoints\*.pth 2>nul | findstr "model_best.pth"
if errorlevel 1 (
    echo [警告] 未找到训练好的模型
    echo.
    echo 请先训练模型！
    pause
    goto :menu
)
echo.
echo 提示: 复制上面的检查点路径
set /p checkpoint="请输入检查点路径: "
if not exist "%checkpoint%" (
    echo [错误] 检查点不存在: %checkpoint%
    pause
    goto :menu
)

echo.
echo 开始预测...
echo 检查点: %checkpoint%
echo.
python src/predict.py --checkpoint "%checkpoint%" --visualize --save_predictions --num_samples 10
echo.
pause
goto :menu

:help
echo.
echo =========================================================
echo                      帮助文档
echo =========================================================
echo.
echo 快速开始指南:
echo   1. 安装依赖: pip install -r requirements.txt
echo   2. 下载数据: 从 HadISST 官网下载数据
echo   3. 测试运行: 选择菜单中的 [1] 快速测试
echo   4. 开始训练: 选择菜单中的 [3] 完整训练
echo.
echo 详细文档:
echo   - 使用指南: README_3DSwinSST.md
echo   - 技术细节: README_SwinTransformer.md
echo   - 代码注释: 每个源文件都有详细中文注释
echo.
echo 常见问题:
echo   Q: 显存不足怎么办？
echo   A: 减小 --batch_size 或增大 --spatial_downsample
echo.
echo   Q: 训练多长时间？
echo   A: Tiny模型约4-6小时（50轮，RTX 3090）
echo.
echo   Q: 如何调整模型大小？
echo   A: 修改 --model_type (tiny/small/base) 或编辑配置文件
echo.
echo 获取帮助:
echo   - 查看代码注释（每行关键代码都有解释）
echo   - 阅读 README_3DSwinSST.md（完整使用指南）
echo   - 参考示例配置文件 configs/config.yaml
echo.
pause
goto :menu

:exit
echo.
echo =========================================================
echo   感谢使用 3D Swin Transformer SST预测系统！
echo =========================================================
echo.
echo 提示: 训练结果保存在 outputs/ 目录
echo       使用 TensorBoard 查看训练过程:
echo         tensorboard --logdir outputs
echo.
pause
exit /b 0
