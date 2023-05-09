import uproot
import numpy as np
import os

# import pandas as pd
# from autogluon.tabular import TabularPredictor
from getS50 import getS50
import multiprocessing

# from CorrdinateTransform import corrdinateYBJ

DataPath = "/data02/"
SavePath = "/home2/hky/github/Gamma_Energy/Exptdata/crabCut_23_05_01"


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getsavefilename(path):
    return os.path.join(SavePath, path[-20:-5] + ".npz")


# def getRaDecOff(i):
#     Interval = 2
#     if i == 20:
#         Ra, Dec = corrdinateYBJ(
#             Exptdata["newtheta"], Exptdata["newphi"], Exptdata["mjd"]
#         )
#     else:
#         Ra, Dec = corrdinateYBJ(
#             Exptdata["newtheta"],
#             Exptdata["newphi"]
#             + (-1) ** (i + 1)
#             * (2 * Interval + Interval * int(i / 2))
#             / np.sin(Exptdata["newtheta"]),
#             Exptdata["mjd"],
#         )
#     return [i, Ra, Dec]


# def RaDecCallback(result):
#     i = result[0]
#     Ra = result[1]
#     Dec = result[2]
#     if i == 20:
#         Exptdata["Ra"], Exptdata["Dec"] = Ra, Dec
#     else:
#         Exptdata[f"RaOff_{i}"], Exptdata[f"DecOff_{i}"] = Ra, Dec


columns_need = [
    "nch",
    "cx",
    "cy",
    "sumpf",
    "summd",
    "mr1",
    "ne",
    "age",
    "sigma",
    "theta",
    "phi",
    "mjd",
]

# predictor_energy = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/log10Energy"
# )
# predictor_deltatheta = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltatheta"
# )
# predictor_deltaphi = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltaphi"
# )
# predictor_gamma_CR = TabularPredictor.load(
#     "/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR"
# )


def cutExptdata(filename):
    savefilename = getsavefilename(filename)
    if os.path.exists(savefilename):
        return
    outputpath_, _ = os.path.split(savefilename)
    mkdir(outputpath_)
    Exptdata = uproot.open(filename)
    Exptdata = Exptdata["asresult"].arrays(columns_need, library="np")
    Exptdata_S50 = getS50(Exptdata["ne"], Exptdata["age"])
    Exptdata_cuted = np.where(
        (Exptdata["age"] < 1.3)
        & (Exptdata["age"] > 0.31)
        & (Exptdata_S50 >= 10**-1.2)
        & (Exptdata["nch"] >= 16)
        & (Exptdata["theta"] < 60)
    )
    Exptdata = {key: Exptdata[key][Exptdata_cuted] for key in Exptdata}
    Exptdata["S50"] = Exptdata_S50[Exptdata_cuted]
    np.savez(
        savefilename,
        **Exptdata,
    )


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=3)

    for root, dirs, files in os.walk(DataPath):
        for name in files:
            filename = os.path.join(root, name)
            pool.apply_async(cutExptdata, args=(filename,))
    pool.close()
    pool.join()
    # savefilename = getsavefilename(filename)
    # outputpath_, _ = os.path.split(savefilename)
    # mkdir(outputpath_)
    # Exptdata = uproot.open(filename)
    # Exptdata = Exptdata["asresult"].arrays(columns_need, library="np")
    # Exptdata_S50 = getS50(Exptdata["ne"], Exptdata["age"])
    # Exptdata_cuted = np.where(
    #     (Exptdata["age"] < 1.3)
    #     & (Exptdata["age"] > 0.31)
    #     & (Exptdata_S50 >= 10**-1.2)
    #     & (Exptdata["nch"] >= 16)
    #     & (Exptdata["theta"] < 60)
    # )
    # Exptdata = {key: Exptdata[key][Exptdata_cuted] for key in Exptdata}
    # Exptdata["S50"] = Exptdata_S50[Exptdata_cuted]
    # # Exptdata_df = pd.DataFrame(Exptdata)
    # # Exptdata["isgamma"] = predictor_gamma_CR.predict_proba(Exptdata_df)[1]
    # # Exptdata_df["sumpf"] = np.log10(Exptdata_df["sumpf"])
    # # Exptdata["energy"] = 10 ** predictor_energy.predict(Exptdata_df)
    # # Exptdata["newtheta"] = Exptdata["theta"] - predictor_deltatheta.predict(
    # #     Exptdata_df
    # # )
    # # Exptdata["newphi"] = Exptdata["phi"] - predictor_deltaphi.predict(
    # #     Exptdata_df
    # # )
    # # Exptdata["newtheta"][Exptdata["newtheta"] < 0] = 0
    # # pool = multiprocessing.Pool(processes=21)
    # # for i in range(21):
    # #     pool.apply_async(getRaDecOff, args=(i,), callback=RaDecCallback)

    # # pool.close()
    # # pool.join()
    # np.savez(
    #     savefilename,
    #     **Exptdata,
    # )
    # # count += 1
    # # print(count)
