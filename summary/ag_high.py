from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd


train_data = np.load("./lowEnergy_trainData.npz")

label = "isgamma"

columns_need = [
    "nch",
    "theta",
    "sigma",
    "cx",
    "cy",
    "sumpf",
    "mr1",
]
columns_need.append(label)
train_data = {key: train_data[key] for key in columns_need}
for para in ["sumpf"]:
    train_data[para] = np.log10(train_data[para])
train_data = pd.DataFrame(train_data)

predictor = TabularPredictor(
    label=label,
    path="./autogloun_model/lowEnergy_Allcolumn",
    eval_metric="roc_auc",
).fit(
    train_data,
    num_cpus=40,
    num_gpus=2,
    # presets="best_quality"
)
# predictor = TabularPredictor.load("./autogloun_model/highEnergy_Allcolumn/")


train_data = np.load("./highEnergy_trainData.npz")

label = "isgamma"

columns_need = [
    "nch",
    "theta",
    #     "phi",
    "sigma",
    "cx",
    "cy",
    "sumpf",
    "summd",
    "mr1",
    "ne",
    #     "age",
    "S50",
]
columns_need.append(label)
train_data = {key: train_data[key] for key in columns_need}
for para in ["ne", "S50", "sumpf"]:
    train_data[para] = np.log10(train_data[para])
train_data = pd.DataFrame(train_data)
predictor = TabularPredictor(
    label=label,
    path="./autogloun_model/highEnergy_fewcolumn_best",
    eval_metric="roc_auc",
).fit(train_data[columns_need], num_cpus=40, num_gpus=2, presets="best_quality")
