from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd


# train_data = np.load(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Datawithe_Galactic_2000_withMC_train_2.npz"
# )

# label = "isgamma"

# columns_need = [
#     # "nch",
#     # "theta",
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
# predictor = TabularPredictor(
#     label=label,
#     path="/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_Allsky_MC_4par_likeExpt",
#     eval_metric="roc_auc",
# ).fit(train_data, num_cpus=30, num_gpus=2)

train_data = np.load(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Datawithe_Galactic_2000_withMC_train_random.npz"
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
    # "mr1",
    # "ne",
    # "age",
    # "S50",
    "isgamma",
]

train_data = pd.DataFrame({key: train_data[key] for key in columns_need})
# train_data["cx"] = np.abs(train_data["cx"])
# train_data["cy"] = np.abs(train_data["cy"])
predictor = TabularPredictor(
    label=label,
    path="/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_Allsky_MC_5par_random_2",
    eval_metric="roc_auc",
).fit(train_data, num_cpus=40, num_gpus=2)

# train_data = np.load(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Datawithe_Galactic_2000_withMC_train_2.npz"
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
# train_data["sectheta"] = 1 / np.cos(np.deg2rad(train_data["theta"]))
# columns_need.append("sectheta")
# columns_need.remove("theta")
# predictor = TabularPredictor(
#     label=label,
#     path="/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_Allsky_MC_5par_likeExpt_sectheta",
#     eval_metric="roc_auc",
# ).fit(train_data[columns_need], num_cpus=30, num_gpus=2)