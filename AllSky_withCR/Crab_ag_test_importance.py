from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_data = np.load(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/test_Data_part.npz"
)

columns_need = [
    "nch",
    "cx",
    "cy",
    "sumpf",
    "summd",
    "ne",
    "age",
    "S50",
    "sigma",
    "phi",
    "theta",
    "isgamma",
]

test_data = pd.DataFrame({key: test_data[key] for key in columns_need})

predictor = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR"
)
pred = predictor.predict_proba(test_data)
for cut in np.linspace(0.01, 0.99, 99):
    ac = np.sum((pred[1] > cut) & (test_data["isgamma"] == 1)) / np.sum(pred[1] > cut)
    eff = np.sum((pred[1] > cut) & (test_data["isgamma"] == 1)) / np.sum(
        test_data["isgamma"] == 1
    )
    CR = np.sum((pred[1] < cut) & (test_data["isgamma"] == 0)) / np.sum(
        test_data["isgamma"] == 0
    )
    print(f"{cut:.2f}", f"{ac*100:.2f}%", f"{eff*100:.2f}%", f"{CR*100:.2f}%")
