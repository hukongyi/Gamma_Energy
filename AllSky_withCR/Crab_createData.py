import uproot
import os
import numpy as np
from getS50 import getS50
from sklearn.model_selection import train_test_split
from tqdm import tqdm

test_size = 0.4

# paralist_Expt = [
#     "nch",
#     "theta",
#     "phi",
#     "sigma",
#     "cx",
#     "cy",
#     "sumpf",
#     "summd",
#     "mjd",
#     "mr1",
#     "ne",
#     "age",
#     "S50",
#     "Ra",
#     "Dec",
# ]


# CRdata = dict()
# for key in paralist_Expt:
#     CRdata[key] = list()
# count = 0
# datalist = list()
# for root, dirs, files in os.walk(
#     "/home2/hky/github/Gamma_Energy/Exptdata/crabcutData_eqzenith"
# ):
#     for name in files:
#         datalist.append(os.path.join(root, name))
# for path in datalist:
#     Exptdata = np.load(path)
#     Exptcut = np.where(
#         (Exptdata["Dec"] < 23)
#         & (Exptdata["Dec"] > 21)
#         & (
#             (Exptdata["summd"] < 5.1e-3 * Exptdata["sumpf"] ** 1.2)
#             | (Exptdata["summd"] < 0.4)
#         )
#     )
#     for key in paralist_Expt:
#         CRdata[key].append(Exptdata[key][Exptcut])

# for key in paralist_Expt:
#     CRdata[key] = np.concatenate(CRdata[key])
# CRdata["isgamma"] = np.zeros_like(CRdata["nch"])
# CrabData = np.load("/home2/hky/github/Gamma_Energy/Exptdata/mergedData_nearcrab.npz")
# CrabDatacuted = np.where(
#     (CrabData["summd"] < 5.1e-3 * CrabData["sumpf"] ** 1.2) | (CrabData["summd"] < 0.4)
# )

# CrabData = {key: CrabData[key][CrabDatacuted] for key in CrabData}
# CrabData["isgamma"] = np.ones_like(CrabData["nch"])
# paralist_Expt.append("isgamma")
# for key in paralist_Expt:
#     CRdata[key] = np.concatenate([CRdata[key], CrabData[key]])

data = np.load(
    "/home2/hky/github/Gamma_Energy/Exptdata/mergedData_eqzenith_mdcut.npz"
)
data = {key: data[key] for key in data}
data["isgamma"][
    np.where(
        (data["Dec"] < 22.5)
        & (data["Dec"] > 21.5)
        & (data["Ra"] < 84.1)
        & (data["Ra"] > 83.1)
    )
]=1
data["isgamma"][
    np.where(
        (data["Dec"] < 37)
        & (data["Dec"] > 36)
        & (data["Ra"] < 305.25)
        & (data["Ra"] > 304.25)
    )
]=1
data["isgamma"][
    np.where(
        (data["Dec"] < 61.5)
        & (data["Dec"] > 60.5)
        & (data["Ra"] < 337.25)
        & (data["Ra"] > 336.25)
    )
]=1
data["isgamma"][
    np.where(
        (data["Dec"] < 6.5)
        & (data["Dec"] > 5.5)
        & (data["Ra"] < 287.5)
        & (data["Ra"] > 286.5)
    )
]=1
data["isgamma"][
    np.where(
        (data["Dec"] < 41.5)
        & (data["Dec"] > 40.5)
        & (data["Ra"] < 308.25)
        & (data["Ra"] > 307.25)
    )
]=1
data["isgamma"][
    np.where(
        (data["Dec"] < -12.5)
        & (data["Dec"] > -14)
        & (data["Ra"] < 277)
        & (data["Ra"] > 276)
    )
]=1
# data["isgamma"] = np.where((data["Ra"] < 84.1) & (data["Ra"] > 83.1), 1, 0)
train_index, test_index = train_test_split(
    range(len(data["nch"])), test_size=test_size, random_state=42
)
CRdata_train = {key: data[key][train_index] for key in data}
CRdata_test = {key: data[key][test_index] for key in data}
np.savez(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Crab_Dec_train_mulitysource.npz",
    **CRdata_train
)
np.savez(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Crab_Dec_test_mulitysource.npz", **CRdata_test
)
# CRdata["isgamma"] = np.zeros_like(CRdata["nch"])


# MCcut = np.where(
#     (gammadata["inout"] == 1)
#     & (gammadata["age"] < 1.3)
#     & (gammadata["age"] > 0.31)
#     & (gammadata["S50"] >= 10**-1.2)
#     & (gammadata["nch"] >= 16)
#     & (gammadata["theta"] < 60)
# )
# print(MCcut[0].shape)
# Alldata = {
#     key: np.concatenate([gammadata[key][MCcut], CRdata[key]]) for key in paralist_All
# }

# train_index, test_index = train_test_split(
#     range(len(Alldata["nch"])), test_size=0.4, random_state=42
# )
# traindata = {key: Alldata[key][train_index] for key in Alldata.keys()}
# testdata = {key: Alldata[key][test_index] for key in Alldata.keys()}
# np.savez(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/train_Data_Allsky.npz",
#     **traindata
# )
# np.savez(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/test_Data_Allsky.npz", **testdata
# )
