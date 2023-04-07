import numpy as np
import h5py

# 加载npz文件
data = np.load("/home2/hky/github/Gamma_Energy/AllSky_orginData/Data/gamma_allsky.npz")

# 创建h5文件
f = h5py.File(
    "/home2/hky/github/Gamma_Energy/AllSky_orginData/Data/gamma_allsky.h5", "w"
)

# 遍历npz文件中的每个数组
for key in data.files:
    print(key)
    # 保存数组到h5文件中，使用相同的名字
    f.create_dataset(key, data=data[key])

# 关闭h5文件
f.close()
