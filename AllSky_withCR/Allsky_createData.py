from CorrdinateTransform import corrdinateYBJGalactic
import ctypes
import os
import numpy as np
import multiprocessing
import math
from tqdm import tqdm
import random


def getGalactic(shared_array, order):
    theta = shared_array[0]
    phi = shared_array[1]
    mjd = shared_array[2]
    Galacticl, Galacticb = corrdinateYBJGalactic(theta, phi, mjd)
    return [order, Galacticl, Galacticb]


def GalacticCallback(result):
    order = result[0]
    Galacticl = result[1]
    Galacticb = result[2]
    pbar.update(1)
    AllskyData["l"][order], AllskyData["b"][order] = Galacticl, Galacticb


def errorCallBack(error):
    print(error)


paralist_Expt = [
    "nch",
    "theta",
    "phi",
    "sigma",
    "cx",
    "cy",
    "sumpf",
    "summd",
    "mr1",
    "ne",
    "age",
    "mjd",
    "S50",
]

AllskyData = dict()
for key in paralist_Expt:
    AllskyData[key] = list()

datalist = list()
for root, dirs, files in os.walk(
    "/home2/hky/github/Gamma_Energy/Exptdata/crabCut_23_05_01", topdown=False
):
    for name in files:
        datalist.append(os.path.join(root, name))

# sumpfmin = np.array([30,100, 200, 400, 800, 1000, 2000])
# count1 = np.zeros_like(sumpfmin)
# count2 = np.zeros_like(sumpfmin)
sample_num = 2000
count = 0
for path in random.sample(datalist,sample_num):
    Exptdata = np.load(path)
        
    Exptdata_cut = np.where(
        (Exptdata["summd"] < 0.4)
        | (Exptdata["summd"] < 5.1e-3 * Exptdata["sumpf"] ** 1.2)
    )

    for key in paralist_Expt:
        AllskyData[key].append(Exptdata[key][Exptdata_cut])
for key in paralist_Expt:
    AllskyData[key] = np.concatenate(AllskyData[key])
# # for path in random.sample(datalist, sample_num):
# #     print(path)
# #     Exptdata = np.load(path)
# #     for key in paralist_Expt:
# #         AllskyData[key].append(Exptdata[key])
# # for key in paralist_Expt:
# #     AllskyData[key] = np.concatenate(AllskyData[key])

np.savez(
    f"/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Data_{sample_num}_random.npz",
    **AllskyData,
)
AllskyData = np.load(
    f"/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Data_{sample_num}_random.npz"
)

AllskyData = {key: AllskyData[key] for key in AllskyData}
AllskyData["l"] = np.zeros_like(AllskyData["theta"])
AllskyData["b"] = np.zeros_like(AllskyData["theta"])
shared_array_base = multiprocessing.Array(ctypes.c_double, 3 * len(AllskyData["theta"]))
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(3, len(AllskyData["theta"]))
shared_array[0, :] = AllskyData["theta"][:]
shared_array[1, :] = AllskyData["phi"][:]
shared_array[2, :] = AllskyData["mjd"][:]
order = np.array_split(
    np.arange(len(AllskyData["theta"])), math.ceil(len(AllskyData["theta"]) / 1e5)
)
pbar = tqdm(total=len(order))

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=20)
    for order_tmp in order:
        pool.apply_async(
            getGalactic,
            args=(shared_array[:, order_tmp], order_tmp),
            callback=GalacticCallback,
            error_callback=errorCallBack,
        )
    pool.close()
    pool.join()

    np.savez(
        f"/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Datawithe_Galactic_{sample_num}_random.npz",
        **AllskyData,
    )
