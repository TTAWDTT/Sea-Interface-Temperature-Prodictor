# Sea-Interface-Temperature-Prodictor

预测海平面的整体温度变化，并输出图片

## tips

先去 [https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html](https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html) 中下载HadISST_sst.nc.gz数据集并解压后放到仓库根目录。

当前提供scripts下的两个脚本：

- scripts\readnc.py可用于读取.nc文件原文
- scripts\show_data.py可用于可视化呈现数据，提供GIF、序列图线、IMAGE三类可视化，在scripts\README.md中有使用说明

