from autogluon.tabular import TabularDataset, TabularPredictor
# import uproot
import os
import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
from ALLSKY_energy_function import *
from getS50 import *
# from draw_compare_multiply import draw_compare_multiply
# from sklearn.preprocessing import PowerTransformer


# file = uproot.open("/home2/chenxu/data/gamma_all.root")
# test_size = 0.4

# data = file["asresult"].arrays(["nch", "theta", "phi", "sigma", "cx", "cy", "sumpf",
#                                "summd", "mr1", "ne", "age", "pritheta", "priphi", "prie", "inout"], library="np")

# data["cr"] = np.sqrt(data["cx"]**2+data["cy"]**2)
# nch = data["nch"]
# theta = data["theta"]
# phi = data["phi"]
# sigma = data["sigma"]
# cx = data["cx"]
# cy = data["cy"]
# cr = data["cr"]
# sumpf = data["sumpf"]
# summd = data["summd"]
# mr1 = data["mr1"]
# ne = data["ne"]
# age = data["age"]
# pritheta = np.rad2deg(data["pritheta"])
# priphi = 180-np.rad2deg(data["priphi"])
# priphi[priphi > 180] = priphi[priphi > 180]-360
# prie = data["prie"]
# inout = data["inout"]
# S50 = getS50(ne, age)


# cuted = np.where((theta < 60) & (nch >= 16) & (
#     inout == 1) & (age > 0.31) & (age < 1.59) & (sigma < 1) & (ne > 1e4))

# nch = nch[cuted]
# theta = theta[cuted]
# phi = phi[cuted]
# sigma = sigma[cuted]
# cx = cx[cuted]
# cy = cy[cuted]
# sumpf = sumpf[cuted]
# summd = summd[cuted]
# mr1 = mr1[cuted]
# ne = ne[cuted]
# age = age[cuted]
# pritheta = pritheta[cuted]
# priphi = priphi[cuted]
# prie = prie[cuted]/1000
# inout = inout[cuted]
# S50 = S50[cuted]
# sectheta = 1/np.cos(np.deg2rad(theta))

# train_index, test_index = train_test_split(
#     range(nch.shape[0]), test_size=test_size, shuffle=True, random_state=42)

# nch_train = nch[train_index]
# theta_train = theta[train_index]
# phi_train = phi[train_index]
# sigma_train = sigma[train_index]
# cx_train = cx[train_index]
# cy_train = cy[train_index]
# sumpf_train = sumpf[train_index]
# summd_train = summd[train_index]
# mr1_train = mr1[train_index]
# ne_train = ne[train_index]
# age_train = age[train_index]
# pritheta_train = pritheta[train_index]
# priphi_train = priphi[train_index]
# prie_train = prie[train_index]
# inout_train = inout[train_index]
# S50_train = S50[train_index]
# sectheta_train = sectheta[train_index]

# nch_test = nch[test_index]
# theta_test = theta[test_index]
# phi_test = phi[test_index]
# sigma_test = sigma[test_index]
# cx_test = cx[test_index]
# cy_test = cy[test_index]
# sumpf_test = sumpf[test_index]
# summd_test = summd[test_index]
# mr1_test = mr1[test_index]
# ne_test = ne[test_index]
# age_test = age[test_index]
# pritheta_test = pritheta[test_index]
# priphi_test = priphi[test_index]
# prie_test = prie[test_index]
# inout_test = inout[test_index]
# S50_test = S50[test_index]
# sectheta_test = sectheta[test_index]


# sc = PowerTransformer()
# x = np.array([
#     S50_train,
#     sectheta_train,
#     nch_train,
#     sumpf_train,
#     summd_train,
#     mr1_train,
#     ne_train,
#     age_train,
#     cx_train,
#     cy_train,
#     sigma_train,
# ]).T
# y = np.log10(prie_train).reshape(-1, 1)

# X = sc.fit_transform(x)
# name_list = ["S50", "sectheta", "nch", "sumpf",
#              "summd", "mr1", "ne", "age", "cx", "cy", "sigma", "log_energy"]

# pd_data = pd.DataFrame(np.concatenate([X, y], axis=1), columns=name_list)

# pd_data.to_csv(
#     "/home2/hky/github/Gamma_Energy/AllSky/MC_train_AllSky_Data_transformed.csv", index=False)

# X_test = np.array([
#     S50_test,
#     sectheta_test,
#     nch_test,
#     sumpf_test,
#     summd_test,
#     mr1_test,
#     ne_test,
#     age_test,
#     cx_test,
#     cy_test,
#     sigma_test,
# ]).T
# X_test = sc.transform(X_test)
# y_test = np.log10(prie_test).reshape(-1, 1)
# pd_data = pd.DataFrame(np.concatenate(
#     [X_test, y_test], axis=1), columns=name_list)

# pd_data.to_csv(
#     "/home2/hky/github/Gamma_Energy/AllSky/MC_test_AllSky_Data_transformed.csv", index=False)

savepath = "/home2/hky/github/Gamma_Energy/AllSky/fig/energy_reconstruction/"
method = "autogluon"
tmpsavepath = os.path.join(savepath, method)
mkdir(tmpsavepath)


train_data_autogluon = TabularDataset(
    "/home2/hky/github/Gamma_Energy/AllSky/MC_train_AllSky_Data_transformed.csv")
time_limit = 24*60*60
predictor = TabularPredictor(label="log_energy").fit(
    train_data_autogluon, time_limit=time_limit, num_gpus=2, presets="high_quality")
# predictor = TabularPredictor.load("/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/ag-20230327_164904")
test_data_autogluon = TabularDataset(
    "/home2/hky/github/Gamma_Energy/AllSky/MC_test_AllSky_Data_transformed.csv")
energy_pred = 10**predictor.predict(test_data_autogluon.drop(
    columns=["log_energy"])).to_numpy()
energy_orgin = 10**test_data_autogluon["log_energy"].to_numpy()

check_fit(energy_pred, energy_orgin, method, tmpsavepath)

# draw_precision()
np.save("/home2/hky/github/Gamma_Energy/AllSky/precision_ALLSKY_sigma<1_ne>1e4.npy", precision)
