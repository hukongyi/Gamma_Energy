import numpy as np
from autogluon.tabular import TabularPredictor
import pandas as pd
from tqdm import tqdm
import os


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


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getsavefilename(SavePath, path):
    return os.path.join(SavePath, path[-19:-4] + ".npz")


DataPath = "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17"
SavePath = "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17_merged"


mkdir(SavePath)

for root, dirs, _ in os.walk(DataPath):
    for dirname in dirs:
        Exptdata = dict()
        for key in columns_need:
            Exptdata[key] = list()
        for root2, _, files in os.walk(os.path.join(root, dirname)):
            for filename in files:
                Exptdatatmp = np.load(os.path.join(root2, filename))
                for key in Exptdatatmp:
                    Exptdata[key].append(Exptdatatmp[key])
        for key in Exptdata.keys():
            Exptdata[key] = np.concatenate(Exptdata[key])
        np.savez(os.path.join(SavePath, dirname) + ".npz", **Exptdata)
