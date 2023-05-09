from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd


train_data = np.load(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Datawithe_Galactic_1000_train.npz"
)

label = "isgamma"

columns_need = [
    "nch",
    "theta",
    # "phi",
    "sigma",
    "cx",
    "cy",
    "sumpf",
    "summd",
    "mr1",
    "ne",
    "age",
    "S50",
    "isgamma",
]
train_data = pd.DataFrame({key: train_data[key] for key in columns_need})
predictor = TabularPredictor(
    label=label,
    path="/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_Allsky_AllExpt_11par",
    eval_metric="roc_auc",
).fit(train_data, num_cpus=30, num_gpus=2)
