import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import pandas as pd


data = np.load(
    "/home2/hky/github/Gamma_Energy/AllSky_orginData/Data/test_cuted_data.npz"
)
train_index, test_index = train_test_split(
    range(len(data["nch"])), random_state=42, test_size=8 / 9
)
data = {key: data[key][train_index] for key in data}


Predictor_log10energy = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky_orginData/agmodel/log10Energy_orgindata/"
)
Predictor_deltatheta = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky_orginData/agmodel/deltatheta/"
)
Predictor_deltaphi = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky_orginData/agmodel/deltaphi/"
)

data_df = pd.DataFrame(data)

print("energy")
data["pred_log10energy"] = Predictor_log10energy.predict(data_df).to_numpy()

print("deltatheta")
data["pred_deltatheta"] = Predictor_deltatheta.predict(data_df).to_numpy()
print("deltaphi")
data["pred_deltaphi"] = Predictor_deltaphi.predict(data_df).to_numpy()
del data_df

loc = np.loadtxt("/home2/hky/github/Gamma_Energy/AllSky_orginData/TibetIII-forplot.loc")

x = list()
y = list()
for i in range(996):
    if np.max(data[f"Tibet_pf{i}"]) != 0:
        x.append(loc[i, 3])
        y.append(loc[i, 4])
x = np.array(x)
y = np.array(y)

x_set = list(set(x))
x_set.sort()
y_set = list(set(y))
y_set.sort()

x_dict = dict()
y_dict = dict()
for i in range(len(x_set)):
    x_dict[x_set[i]] = i + 1
    y_dict[y_set[i]] = i + 1

data["Matrix"] = np.zeros([len(data["nch"]), 2, 32, 32])

for j in range(len(data["nch"])):
    for i in range(996):
        if loc[i, 3] in x_dict.keys() and loc[i, 4] in y_dict.keys():
            data["Matrix"][j, 0, x_dict[loc[i, 3]], y_dict[loc[i, 4]]] = data[
                f"Tibet_pf{i}"
            ][j]
            data["Matrix"][j, 1, x_dict[loc[i, 3]], y_dict[loc[i, 4]]] = data[
                f"Tibet_T{i}"
            ][j]
np.savez(
    "/home2/hky/github/Gamma_Energy/AllSky_orginData/Data/train_cuted_data_CNN.npz",
    **data,
)
