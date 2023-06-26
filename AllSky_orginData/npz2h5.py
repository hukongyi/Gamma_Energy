import numpy as np
import h5py

# 加载npz文件
data = np.load("/home2/hky/github/Gamma_Energy/AllSky_orginData/Data/Comsic_allsky.npz")

cuted = np.where((data['theta'][:] < 60) & (
    data['inout'][:] == 1) & (data['age'][:] > 0.31) & (data['age'][:] < 1.59) & (data['sigma'][:] < 1.) & (data['ne'][:] > 1e4)&(data["oldtrig"]&0x1==1))

# 创建h5文件
f_compress = h5py.File(
    "/home2/hky/github/Gamma_Energy/AllSky_orginData/Data/Comsic_allsky_compress_cuted.h5", "w"
)

for key in data.keys():
    print(key)
    tmp = data[key][:]
    # 保存数组到h5文件中，使用相同的名字
    f_compress.create_dataset(key, data=tmp[cuted], chunks=True, compression="gzip")

# # 遍历npz文件中的每个数组
# for key in data.files:
#     print(key)
#     # 保存数组到h5文件中，使用相同的名字
#     f_compress.create_dataset(key, data=data[key])

# 关闭h5文件
f_compress.close()
