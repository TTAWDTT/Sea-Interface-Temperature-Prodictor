import xarray as xr
import numpy as np

ds = xr.open_dataset("HadISST_sst.nc", decode_times=True)
print(ds)

# 掩膜异常哨兵值
sst = ds["sst"].where(ds["sst"] > -100)

print("time_start:", np.datetime_as_string(ds["time"].values[0], unit="D"))
print("time_end:", np.datetime_as_string(ds["time"].values[-1], unit="D"))
print("shape:", sst.shape)
print("min:", float(sst.min(skipna=True)))
print("max:", float(sst.max(skipna=True)))
print("mean:", float(sst.mean(skipna=True)))