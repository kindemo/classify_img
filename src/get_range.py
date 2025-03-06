import numpy as np
# 根据头文件信息定义数据形状和类型
dim_size = (512, 512, 121)  # DimSize = 512 512 121
data_type = np.int16        # MET_SHORT对应int16
# 读取.raw文件
with open("D:/BaiduNetdiskDownload/LUNA16/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.raw", "rb") as f:
    ct_data = np.fromfile(f, dtype=data_type).reshape(dim_size)

min_val = np.min(ct_data)
max_val = np.max(ct_data)
print(f"CT值范围: [{min_val}, {max_val}]")