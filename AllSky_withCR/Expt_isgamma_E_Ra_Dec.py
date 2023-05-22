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
            + (-1) ** (i + 1)
            * (2 * Interval + Interval * int(i / 2))
            / np.sin(np.deg2rad(theta)),
            mjd,
        )
    return [i, order, Ra, Dec]


def RaDecCallback(result):
    pbar.update(1)
    i = result[0]
    order = result[1]
    Ra = result[2]
    Dec = result[3]
    if i == 20:
        Exptdata["Ra"][order], Exptdata["Dec"][order] = Ra, Dec
    else:
        Exptdata[f"RaOff_{i}"][order], Exptdata[f"DecOff_{i}"][order] = Ra, Dec


DataPath = "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17/cutedData.npz"
SavePath1 = (
    "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17/cutedData_isgamma.npz"
)
# SavePath2 = (
#     "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17/cutedData_E_isgamma.npz"
# )
# SavePath3 = (
#     "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17/cutedData_E_RaDec.npz"
# )
# Exptdata = np.load(DataPath)
# Exptdata = {key: Exptdata[key] for key in Exptdata}
# predictor_isgamma = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_Allsky_MC_5par_random_2/"
# )
# Exptdata_df = pd.DataFrame(Exptdata)
# Exptdata["isgamma"] = predictor_isgamma.predict_proba(pd.DataFrame(Exptdata_df))[
#     1
# ].to_numpy()

# np.savez(
#     SavePath1,
#     **Exptdata,
# )
Exptdata = np.load(SavePath1)
Exptdata = {key: Exptdata[key] for key in Exptdata}

P_value = np.array([0.36, 0.44, 0.55, 0.74, 0.91, 0.98, 0.99])


# predictor_energy = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/log10Energy"
# )
# predictor_deltatheta = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltatheta"
# )
# predictor_deltaphi = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltaphi"
# )


# Exptdata_df = pd.DataFrame(Exptdata)
# Exptdata_df["sumpf"] = np.log10(Exptdata_df["sumpf"])
# print("energy")
# Exptdata["energy"] = 10 ** predictor_energy.predict(Exptdata_df).to_numpy()
# print("newtheta")
# Exptdata["newtheta"] = (
#     Exptdata["theta"] - predictor_deltatheta.predict(Exptdata_df).to_numpy()
# )

# print("newphi")
# Exptdata["newphi"] = (
#     Exptdata["phi"] - predictor_deltaphi.predict(Exptdata_df).to_numpy()
# )

# Exptdata["newtheta"][Exptdata["newtheta"] < 0] = 0.01
# np.savez(
#     SavePath1,
#     **Exptdata,
# )
# Exptdata_df = pd.DataFrame(Exptdata)

# for para_num in [7, 6, 5, 4]:
#     print(para_num)
#     predictor_gamma_CR = TabularPredictor.load(
#         f"/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_{para_num}par/"
#     )
#     Exptdata[f"isgamma_{para_num}"] = predictor_gamma_CR.predict(Exptdata_df).to_numpy()

# np.savez(
#     SavePath2,
#     **Exptdata,
# )
# pbar = tqdm(total=math.ceil(len(Exptdata["newtheta"]) / 3e4) * 21)

# shared_array_base = multiprocessing.Array(
#     ctypes.c_double, 3 * len(Exptdata["newtheta"])
# )
# Exptdata["Ra"] = np.zeros_like(Exptdata["newtheta"])
# Exptdata["Dec"] = np.zeros_like(Exptdata["newtheta"])
# for i in range(20):
#     Exptdata[f"RaOff_{i}"] = np.zeros_like(Exptdata["newtheta"])
#     Exptdata[f"DecOff_{i}"] = np.zeros_like(Exptdata["newtheta"])

# shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
# shared_array = shared_array.reshape(3, len(Exptdata["newtheta"]))
# shared_array[0, :] = Exptdata["newtheta"][:]
# shared_array[1, :] = Exptdata["newphi"][:]
# shared_array[2, :] = Exptdata["mjd"][:]
# order = np.array_split(
#     np.arange(len(Exptdata["newtheta"])), math.ceil(len(Exptdata["newtheta"]) / 3e4)
# )
# if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=20)

#     for i in range(21):
#         for order_tmp in order:
#             pool.apply_async(
#                 getRaDecOff,
#                 args=(i, shared_array[:, order_tmp], order_tmp),
#                 callback=RaDecCallback,
#             )
#     pool.close()
#     pool.join()
#     np.savez(
#         SavePath3,
#         **Exptdata,
#     )
