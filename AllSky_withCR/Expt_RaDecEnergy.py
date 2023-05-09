# import uproot
import numpy as np
import os

import pandas as pd
from autogluon.tabular import TabularPredictor
import math

from getS50 import getS50
import multiprocessing
import ctypes
from CorrdinateTransform import corrdinateYBJ
from tqdm import tqdm


DataPath = "/home2/hky/github/Gamma_Energy/Exptdata/crabCut_23_05_01"
SavePath = "/home2/hky/github/Gamma_Energy/Exptdata/crabCut_23_05_03_withisgamma_Allsky/Data.npz"
SavePath2 = "/home2/hky/github/Gamma_Energy/Exptdata/crabCut_23_05_03_withisgamma_Allsky/Data_withenergy.npz"
SavePath3 = "/home2/hky/github/Gamma_Energy/Exptdata/crabCut_23_05_03_withisgamma_Allsky/Data_withRaDec.npz"


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getsavefilename(path):
    return os.path.join(SavePath, path[-19:-4] + ".npz")


def getRaDecOff(i, shared_array, order):
    theta = shared_array[0]
    phi = shared_array[1]
    mjd = shared_array[2]
    Interval = 2
    if i == 20:
        Ra, Dec = corrdinateYBJ(theta, phi, mjd)
    else:
        Ra, Dec = corrdinateYBJ(
            theta,
            phi
            + (-1) ** (i + 1) * (2 * Interval + Interval * int(i / 2)) / np.sin(theta),
            mjd,
        )
    return [i, order, Ra, Dec]


def RaDecCallback(result):
    i = result[0]
    order = result[1]
    Ra = result[2]
    Dec = result[3]
    if i == 20:
        Exptdata["Ra"][order], Exptdata["Dec"][order] = Ra, Dec
    else:
        Exptdata[f"RaOff_{i}"][order], Exptdata[f"DecOff_{i}"][order] = Ra, Dec


def getenergythetaphi():
    predictor_energy = TabularPredictor.load(
        "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/log10Energy"
    )
    predictor_deltatheta = TabularPredictor.load(
        "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltatheta"
    )
    predictor_deltaphi = TabularPredictor.load(
        "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltaphi"
    )
    Exptdata = dict()

    columns_need = [
        "nch",
        "cx",
        "cy",
        "sumpf",
        "summd",
        "mr1",
        "ne",
        "age",
        "sigma",
        "theta",
        "phi",
        "mjd",
        "S50",
    ]
    for key in columns_need:
        Exptdata[key] = list()
    count = 0
    for root, dirs, files in os.walk(DataPath):
        for name in files:
            filename = os.path.join(root, name)
            Exptdatatmp = np.load(filename)
            Exptdatacut = np.where(Exptdatatmp["isgamma"] > 0.1)
            for key in Exptdatatmp:
                Exptdata[key].append(Exptdatatmp[key][Exptdatacut])
            count += 1
            print(count)
    for key in Exptdata.keys():
        Exptdata[key] = np.concatenate(Exptdata[key])
    np.savez(
        SavePath,
        **Exptdata,
    )
    Exptdata = np.load(SavePath)
    Exptdata = {key: Exptdata[key] for key in Exptdata}
    Exptdata_df = pd.DataFrame(Exptdata)

    Exptdata_df["sumpf"] = np.log10(Exptdata_df["sumpf"])
    print("energy")
    Exptdata["energy"] = 10 ** predictor_energy.predict(Exptdata_df).to_numpy()
    Exptdata["newtheta"] = (
        Exptdata["theta"] - predictor_deltatheta.predict(Exptdata_df).to_numpy()
    )

    print("newtheta")
    Exptdata["newphi"] = (
        Exptdata["phi"] - predictor_deltaphi.predict(Exptdata_df).to_numpy()
    )

    print("newphi")
    Exptdata["newtheta"][Exptdata["newtheta"] < 0] = 0.01
    np.savez(
        SavePath2,
        **Exptdata,
    )


getenergythetaphi()
Exptdata = np.load(SavePath2)
print(Exptdata["sumpf"].shape)
Exptdata = {key: Exptdata[key] for key in Exptdata}
shared_array_base = multiprocessing.Array(
    ctypes.c_double, 3 * len(Exptdata["newtheta"])
)

Exptdata["Ra"] = np.zeros_like(Exptdata["newtheta"])
Exptdata["Dec"] = np.zeros_like(Exptdata["newtheta"])
for i in range(20):
    Exptdata[f"RaOff_{i}"] = np.zeros_like(Exptdata["newtheta"])
    Exptdata[f"DecOff_{i}"] = np.zeros_like(Exptdata["newtheta"])

shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(3, len(Exptdata["newtheta"]))
shared_array[0, :] = Exptdata["newtheta"][:]
shared_array[1, :] = Exptdata["newphi"][:]
shared_array[2, :] = Exptdata["mjd"][:]
order = np.array_split(
    np.arange(len(Exptdata["newtheta"])), math.ceil(len(Exptdata["newtheta"]) / 1e5)
)
print(len(order) * 21)


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=40)

    for i in range(21):
        for order_tmp in order:
            pool.apply_async(
                getRaDecOff,
                args=(i, shared_array[:, order_tmp], order_tmp),
                callback=RaDecCallback,
            )
    pool.close()
    pool.join()
    np.savez(
        SavePath3,
        **Exptdata,
    )
