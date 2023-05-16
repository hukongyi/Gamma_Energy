import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

Crab = np.load("/home2/hky/github/Gamma_Energy/Exptdata/mergedData_eqzenith_mdcut.npz")


Crabcut = np.where(
    (Crab["Dec"] < 22.5)
    & (Crab["Dec"] > 21.5)
    & ((Crab["summd"] < 5e-3 * Crab["sumpf"] ** 1.2) | (Crab["summd"] < 0.4))
)
Crabdata = {key: Crab[key][Crabcut] for key in Crab}

train_index, test_index = train_test_split(
    range(len(Crabdata["Ra"])), test_size=0.4, random_state=42
)

label = "isgamma"
paraneed = {"isgamma", "sumpf", "summd", "cx", "cy"}

predictor = TabularPredictor(label=label, eval_metric="roc_auc").fit(
    pd.DataFrame(Crabdata)[paraneed].loc[train_index],
    num_cpus=30,
    num_gpus=2,
    presets="best_quality",
)
