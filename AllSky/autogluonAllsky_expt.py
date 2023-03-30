import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import uproot
from getS50 import *
from draw_compare_multiply import draw_compare_multiply
from sklearn.preprocessing import PowerTransformer
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

test_size = 0.4


file = uproot.open("/home2/chenxu/data/gamma.00339651.root")
data = file["asresult"].arrays(["nch", "theta", "phi", "sigma", "cx", "cy", "sumpf",
                               "summd", "mr1", "ne", "age", "pritheta", "priphi", "prie", "inout"], library="np")
data["cr"] = np.sqrt(data["cx"]**2+data["cy"]**2)
data["log10Energy"] = np.log10(data["prie"]/1000)
data["pritheta"] = np.rad2deg(data["pritheta"])
data["priphi"] = 180-np.rad2deg(data["priphi"])
data["priphi"][data["priphi"] > 180] = data["priphi"][data["priphi"] > 180]-360
data["S50"] = getS50(data["ne"], data["age"])
data["deltatheta"] = data["theta"]-data["pritheta"]
data["deltaphi"] = data["phi"]-data["priphi"]
data["deltaphi"][data["deltaphi"] < -180] += 360
data["deltaphi"][data["deltaphi"] > 180] -= 360


for if_cut in [0, 1]:
    if if_cut:
        cuted = np.where((data["theta"] < 50) & (data["nch"] >= 16)
                         & (data["inout"] == 1) & (data['sigma'] < 1.)
                         & (data["age"] > 0.31) & (data['age'] < 1.3)
                         & (data["S50"] > 10**-1.2)
                         & (data['sumpf'] > 200)
                         & ((data["summd"] < 1.8e-5*data["sumpf"]**1.8) | (data["summd"] < 0.4)))
    else:
        cuted = np.where(data["inout"] == 1)

    train_index, test_index = train_test_split(
        range(cuted[0].shape[0]), test_size=0.3, random_state=42, shuffle=True)

    data_train = {key: data[key][cuted][train_index] for key in data.keys()}
    data_test = {key: data[key][cuted][test_index] for key in data.keys()}

    pd_data = pd.DataFrame(data_train)
    pd_data_test = pd.DataFrame(data_test)
    pd_data[["sumpf"]] = np.log10(pd_data[["sumpf"]])
    pd_data_test[["sumpf"]] = np.log10(pd_data_test[["sumpf"]])
    train_data_autogluon = TabularDataset(pd_data)
    test_data_autogluon = TabularDataset(pd_data_test)

    for label in [
            "log10Energy",
            "deltatheta",
            "deltaphi"
    ]:
        columns_need = [label, "nch", "theta", "phi", "sigma",
                        "cx", "cy", "sumpf", "summd", "mr1"]

        columns_drop = list()
        for column in train_data_autogluon.columns:
            if column not in columns_need:
                columns_drop.append(column)
        predictor = TabularPredictor(label=label, path=f"/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut={if_cut}/{label}").fit(
            train_data_autogluon.drop(columns=columns_drop), num_gpus=2)
