import numpy as np
import os

save_path = "/home2/hky/github/Gamma_Energy/Exptdata/RaDecOff"
mergedData = dict()
count = 0
for root, dirs, files in os.walk(save_path, topdown=False):
    for name in files:
        data = np.load(os.path.join(root, name))
        for i in data:
            if count == 0:
                mergedData[i] = list()
            mergedData[i].append(data[i])
        count += 1
        print(count)
for i in mergedData.keys():
    mergedData[i] = np.concatenate(mergedData[i])
np.savez_compressed("/home2/hky/github/Gamma_Energy/Exptdata/RaDecOffmergedData.npz",**mergedData)