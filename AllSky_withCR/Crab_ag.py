from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd


# train_data = np.load(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Crab_Dec_train.npz"
# )

# label = "isgamma"

# columns_need = [
#     # "nch",
#     "theta",
#     # "phi",
#     # "sigma",
#     "cx",
#     "cy",
#     "sumpf",
#     "summd",
#     # "mr1",
#     # "ne",
#     # "age",
#     # "S50",
#     "isgamma",
# ]
# train_data = pd.DataFrame({key: train_data[key] for key in columns_need})
# train_data["weight"] = np.where(train_data["isgamma"] == 1, 1, 0.005)
# predictor = TabularPredictor(
#     label=label,
#     sample_weight="weight",
#     path="/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_5par",
#     eval_metric="roc_auc",
# ).fit(train_data, num_cpus=30, num_gpus=2)

train_data = np.load(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Crab_Dec_train_mulitysource.npz"
)

label = "isgamma"

columns_need = [
    # "nch",
    "theta",
    # "phi",
    # "sigma",
    "cx",
    "cy",
    "sumpf",
    "summd",
    "mjd",
    # "mr1",
    # "ne",
    # "age",
    # "S50",
    "isgamma",
]
train_data = pd.DataFrame({key: train_data[key] for key in columns_need})
train_data["mjd"] = np.round(train_data["mjd"]/30)
# train_data["weight"] = np.where(train_data["isgamma"] == 1, 10, 0.01)
predictor = TabularPredictor(
    label=label,
    # sample_weight="weight",
    path="/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_6par_mulitysource",
    eval_metric="roc_auc",
).fit(train_data, num_cpus=30, num_gpus=2)
# train_data = np.load(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Crab_Dec_train.npz"
# )

# label = "isgamma"

# columns_need = [
#     # "nch",
#     "theta",
#     # "phi",
#     "sigma",
#     "cx",
#     "cy",
#     "sumpf",
#     "summd",
#     # "mr1",
#     # "ne",
#     # "age",
#     # "S50",
#     "isgamma",
# ]
# train_data = pd.DataFrame({key: train_data[key] for key in columns_need})
# train_data["weight"] = np.where(train_data["isgamma"] == 1, 1, 0.005)
# predictor = TabularPredictor(
#     label=label,
#     sample_weight="weight",
#     path="/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_6par",
#     eval_metric="roc_auc",
# ).fit(train_data, num_cpus=30, num_gpus=2)
# train_data = np.load(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Crab_Dec_train.npz"
# )

# label = "isgamma"

# columns_need = [
#     # "nch",
#     "theta",
#     # "phi",
#     "sigma",
#     "cx",
#     "cy",
#     "sumpf",
#     "summd",
#     "mjd",
#     # "mr1",
#     # "ne",
#     # "age",
#     # "S50",
#     "isgamma",
# ]
# train_data = pd.DataFrame({key: train_data[key] for key in columns_need})
# train_data["weight"] = np.where(train_data["isgamma"] == 1, 1, 0.005)
# predictor = TabularPredictor(
#     label=label,
#     sample_weight="weight",
#     path="/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_7par",
#     eval_metric="roc_auc",
# ).fit(train_data, num_cpus=30, num_gpus=2)
