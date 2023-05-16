# import uproot
import numpy as np
import os

import pandas as pd
from autogluon.tabular import TabularPredictor
from tqdm import tqdm
import time

# from getS50 import getS50
# import multiprocessing

from CorrdinateTransform import corrdinateYBJ

DataPath = "/home2/hky/github/Gamma_Energy/Exptdata/crabCut_23_05_01"


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getsavefilename(SavePath, path):
    return os.path.join(SavePath, path[-19:-4] + ".npz")


def getRaDecOff(i):
    Interval = 2
    if i == 20:
        Ra, Dec = corrdinateYBJ(
            Exptdata["newtheta"], Exptdata["newphi"], Exptdata["mjd"]
        )
    else:
        Ra, Dec = corrdinateYBJ(
            Exptdata["newtheta"],
            Exptdata["newphi"]
            + (-1) ** (i + 1)
            * (2 * Interval + Interval * int(i / 2))
            / np.sin(Exptdata["newtheta"]),
            Exptdata["mjd"],
        )
    return [i, Ra, Dec]


def RaDecCallback(result):
    i = result[0]
    Ra = result[1]
    Dec = result[2]
    if i == 20:
        Exptdata["Ra"], Exptdata["Dec"] = Ra, Dec
    else:
        Exptdata[f"RaOff_{i}"], Exptdata[f"DecOff_{i}"] = Ra, Dec


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
]

# predictor_energy = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/log10Energy"
# )
# predictor_deltatheta = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltatheta"
# )
# predictor_deltaphi = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltaphi"
# )
predictor_gamma_CR = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_Allsky_AllExpt_11par/"
)

if __name__ == "__main__":
    # for para_num in [7, 4, 5, 6]:
    #     print(para_num)
        # predictor_gamma_CR = TabularPredictor.load(
        #     f"/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_{para_num}par/"
        # )
    SavePath = f"/home2/hky/github/Gamma_Energy/Exptdata/J1857Cut_23_05_14"

    datalist = list()
    for root, dirs, files in os.walk(DataPath):
        for name in files:
            filename = os.path.join(root, name)
            datalist.append(filename)

    for filename in datalist:
        savefilename = getsavefilename(SavePath, filename)
        if os.path.exists(savefilename):
            continue
        outputpath_, _ = os.path.split(savefilename)
        mkdir(outputpath_)
        
        Exptdata = np.load(filename)
        Exptdatacut = np.where(
            (Exptdata["summd"] < 6.3e-4 * Exptdata["sumpf"] ** 1.6)
            | (Exptdata["summd"] < 0.4)
        )
        Exptdata = {key: Exptdata[key][Exptdatacut] for key in Exptdata}
        # Exptdata_df = pd.DataFrame(Exptdata)
        # Exptdata["isgamma"] = predictor_gamma_CR.predict_proba(Exptdata_df)[
        #     1
        # ].to_numpy()
        np.savez(
            savefilename,
            **Exptdata,
        )
