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

DataPath = "/home2/hky/github/Gamma_Energy/Exptdata/J1857Cut_23_05_14"


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


if __name__ == "__main__":
    SavePath = "/home2/hky/github/Gamma_Energy/Exptdata/J1857Cut_23_05_14/cutedData.npz"

    datalist = list()
    Exptdata = dict()
    for key in columns_need:
        Exptdata[key] = list()
    for root, dirs, files in os.walk(DataPath):
        for name in files:
            filename = os.path.join(root, name)
            datalist.append(filename)

    for filename in datalist:
        Exptdatatmp = np.load(filename)
        # Exptdatacut = np.where(
        #     (Exptdatatmp["summd"] < 5.1e-3 * Exptdatatmp["sumpf"] ** 1.2)
        #     | (Exptdatatmp["summd"] < 0.4)
        # )
        for key in Exptdatatmp:
            Exptdata[key].append(Exptdatatmp[key])

    for key in Exptdata.keys():
        Exptdata[key] = np.concatenate(Exptdata[key])
    np.savez(SavePath, **Exptdata)
