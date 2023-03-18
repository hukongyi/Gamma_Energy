from autogluon.tabular import TabularDataset, TabularPredictor
import uproot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from energy_function import *
from getS50 import *
from draw_compare_multiply import draw_compare_multiply


file = uproot.open("/home2/chenxu/data/gamma_all.root")
test_size = 0.4

data = file["asresult"].arrays(["nch", "theta", "phi", "sigma", "cx", "cy", "sumpf",
                               "summd", "mr1", "ne", "age", "pritheta", "priphi", "prie", "inout"], library="np")

data["cr"] = np.sqrt(data["cx"]**2+data["cy"]**2)
nch = data["nch"]
theta = data["theta"]
phi = data["phi"]
sigma = data["sigma"]
cx = data["cx"]
cy = data["cy"]
cr = data["cr"]
sumpf = data["sumpf"]
summd = data["summd"]
mr1 = data["mr1"]
ne = data["ne"]
age = data["age"]
pritheta = np.rad2deg(data["pritheta"])
priphi = 180-np.rad2deg(data["priphi"])
priphi[priphi > 180] = priphi[priphi > 180]-360
prie = data["prie"]
inout = data["inout"]
S50 = getS50(ne, age)


cuted = np.where((theta < 40) & (nch >= 16) & (
    inout == 1) & (age > 0.31) & (age < 1.59))

nch = nch[cuted]
theta = theta[cuted]
phi = phi[cuted]
sigma = sigma[cuted]
cx = cx[cuted]
cy = cy[cuted]
sumpf = sumpf[cuted]
summd = summd[cuted]
mr1 = mr1[cuted]
ne = ne[cuted]
age = age[cuted]
pritheta = pritheta[cuted]
priphi = priphi[cuted]
prie = prie[cuted]/1000
inout = inout[cuted]
S50 = S50[cuted]
sectheta = 1/np.cos(np.deg2rad(theta))

train_index, test_index = train_test_split(
    range(nch.shape[0]), test_size=test_size, shuffle=True, random_state=42)

nch_train = nch[train_index]
theta_train = theta[train_index]
phi_train = phi[train_index]
sigma_train = sigma[train_index]
cx_train = cx[train_index]
cy_train = cy[train_index]
sumpf_train = sumpf[train_index]
summd_train = summd[train_index]
mr1_train = mr1[train_index]
ne_train = ne[train_index]
age_train = age[train_index]
pritheta_train = pritheta[train_index]
priphi_train = priphi[train_index]
prie_train = prie[train_index]
inout_train = inout[train_index]
S50_train = S50[train_index]
sectheta_train = sectheta[train_index]

nch_test = nch[test_index]
theta_test = theta[test_index]
phi_test = phi[test_index]
sigma_test = sigma[test_index]
cx_test = cx[test_index]
cy_test = cy[test_index]
sumpf_test = sumpf[test_index]
summd_test = summd[test_index]
mr1_test = mr1[test_index]
ne_test = ne[test_index]
age_test = age[test_index]
pritheta_test = pritheta[test_index]
priphi_test = priphi[test_index]
prie_test = prie[test_index]
inout_test = inout[test_index]
S50_test = S50[test_index]
sectheta_test = sectheta[test_index]


train_dict = {"nch": nch_train, "sectheta": sectheta_train, "S50": S50_train, "cx": cx_train, "cy": cy_train,
              "ne": ne_train, "age": age_train, "sumpf": sumpf_train, "summd": summd_train, "mr1": mr1_train, "energy": prie_train, "log_energy": np.log10(prie_train)}
pd_data = pd.DataFrame(train_dict)
pd_data.to_csv("MC_train_AllSky_Data.csv", index=False)

test_dict = {"nch": nch_test, "sectheta": sectheta_test, "S50": S50_test, "cx": cx_test, "cy": cy_test,
             "ne": ne_test, "age": age_test, "sumpf": sumpf_test, "summd": summd_test, "mr1": mr1_test, "energy": prie_test, "log_energy": np.log10(prie_test)}
pd_data = pd.DataFrame(test_dict)
pd_data.to_csv("MC_test_AllSky_Data.csv", index=False)

savepath = "./fig/energy_reconstruction/"
method = "autogluon"
tmpsavepath = os.path.join(savepath, method)
mkdir(tmpsavepath)


train_data_autogluon = TabularDataset("MC_train_AllSky_Data.csv")
time_limit = 24*60*60
predictor = TabularPredictor(label="log_energy").fit(train_data_autogluon.drop(
    columns=["energy"]), time_limit=time_limit, presets='best_quality')

test_data_autogluon = TabularDataset("MC_test_Data.csv")
energy_pred = 10**predictor.predict(test_data_autogluon.drop(
    columns=["energy", "log_energy"])).to_numpy()
energy_orgin = prie_test

check_fit(energy_pred, energy_orgin, method, tmpsavepath)

draw_precision()
np.save("precision.npy", precision)