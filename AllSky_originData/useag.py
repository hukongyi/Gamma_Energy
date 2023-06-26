from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

train_data = np.load(
    "/home2/hky/github/Gamma_Energy/AllSky_orginData/Data/train_cuted_data.npz"
)
train_data = {key: train_data[key] for key in train_data}
train_data = pd.DataFrame(train_data)


for label in ["log10Energy", "deltatheta", "deltaphi"]:
    para_need = [
        "nch",
        "theta",
        "phi",
        "sigma",
        "cx",
        "cy",
        "sumpf",
        "summd",
        "mr1",
        "ne",
        "age",
    ]
    para_need.append(label)
    for i in range(996):
        para_need.append(f"Tibet_pf{i}")
        para_need.append(f"Tibet_T{i}")

    predictor = TabularPredictor(
        label=label,
        path=f"/home2/hky/github/Gamma_Energy/AllSky_orginData/agmodel/{label}_orgindata",
    ).fit(train_data[para_need], num_gpus=2, num_cpus=40)
    
    para_need = [
        "nch",
        "theta",
        "phi",
        "sigma",
        "cx",
        "cy",
        "sumpf",
        "summd",
        "mr1",
        "ne",
        "age",
    ]
    para_need.append(label)

    predictor = TabularPredictor(
        label=label,
        path=f"/home2/hky/github/Gamma_Energy/AllSky_orginData/agmodel/{label}",
    ).fit(train_data[para_need], num_gpus=2, num_cpus=40)