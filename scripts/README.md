# HadISST 可视化脚本说明

本目录提供一个脚本 [show_data.py](show_data.py)，用于读取 HadISST 的 .nc 文件并输出三类图形结果：

- 单时刻海表温度空间图（map）
- 指定时段动画（animate）
- 指定经纬度时间序列图和 CSV（series）

默认输入文件为项目根目录下的 HadISST_sst.nc。

## 1. 环境准备

建议在项目根目录 D:\QinBo 下执行。

安装依赖：

```powershell
python -m pip install xarray netCDF4 numpy matplotlib
```

## 2. 快速开始

在项目根目录执行：

```powershell
python scripts/show_data.py --mode map --time 2000-01
```

输出目录默认为项目根目录下的 outputs。

## 3. 三种模式

### 3.1 map：单时刻空间图

按时间选择（推荐）：

```powershell
python scripts/show_data.py --mode map --time 2000-01
```

按索引选择（0 表示第一个月，-1 表示最后一个月）：

```powershell
python scripts/show_data.py --mode map --index -1
```

自定义输出文件：

```powershell
python scripts/show_data.py --mode map --time 2000-01 --output outputs/sst_map_2000-01.png
```

### 3.2 animate：时间段动画

```powershell
python scripts/show_data.py --mode animate --start 2000-01 --end 2001-12 --fps 5
```

减少帧数（每 2 个月取一帧）：

```powershell
python scripts/show_data.py --mode animate --start 1980-01 --end 1990-12 --step 2 --fps 6
```

说明：

- 默认动画格式是 GIF
- 默认最多输出 240 帧，可用 --max-frames 调整

### 3.3 series：点位时间序列

```powershell
python scripts/show_data.py --mode series --lon 120 --lat 30 --start 1980-01 --end 2020-12 --rolling 12
```

该模式会输出两类文件：

- 时间序列图 PNG
- 同名 CSV（包含 sst 和可选滚动均值列）

关闭滚动均值：

```powershell
python scripts/show_data.py --mode series --lon 120 --lat 30 --rolling 1
```

## 4. 常用参数

- --file: 输入 netCDF 文件路径，默认 HadISST_sst.nc
- --output: 输出图像或动画路径
- --cmap: 色标，默认 turbo
- --vmin / --vmax: 颜色范围，默认 -2 到 35
- --show: 保存后弹出图窗（map/series 模式）
- --start / --end: animate 或 series 模式的时间范围
- --step: animate 模式帧采样步长
- --fps: animate 模式每秒帧数
- --lon / --lat: series 模式点位经纬度
- --rolling: series 模式滚动均值窗口（月）
- --csv-output: series 模式 CSV 输出路径

## 5. 运行位置说明

脚本默认用相对路径读取 HadISST_sst.nc，因此推荐在项目根目录执行命令。

如果你在 scripts 目录内执行，请显式指定文件路径：

```powershell
python show_data.py --mode map --time 2000-01 --file ../HadISST_sst.nc
```

## 6. 数据注意事项

脚本已对明显非物理哨兵值进行掩膜（例如 -1000），以避免统计和绘图被异常值污染。

## 7. 常见报错

1) xarray 无法打开 nc 文件，提示后端依赖缺失

安装 netCDF4：

```powershell
python -m pip install netCDF4
```

2) 动画太慢或文件太大

- 缩短时间范围（--start / --end）
- 增大 step（例如 --step 2 或 --step 3）
- 降低 fps
