import numpy as np
from autogluon.tabular import TabularPredictor
import pandas as pd
from tqdm import tqdm
import os
import math
import multiprocessing
import ctypes
from CorrdinateTransform import corrdinateYBJ


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getsavefilename(SavePath, path):
    return os.path.join(SavePath, os.path.basename(path))


def getRaDecOff(i, shared_array, order):
    theta = shared_array[0]
    phi = shared_array[1]
    mjd = shared_array[2]
    Interval = 2
    if i == 20:
        Ra, Dec = corrdinateYBJ(theta, phi, mjd)
    else:
        Ra, Dec = corrdinateYBJ(
            theta,
            phi
            + (-1) ** (i + 1)
            * (2 * Interval + Interval * int(i / 2))
            / np.sin(np.deg2rad(theta)),
            mjd,
        )
    return [i, order, Ra, Dec]


def RaDecCallback(result):
    # pbar.update(1)
    i = result[0]
    order = result[1]
    Ra = result[2]
    Dec = result[3]
    if i == 20:
        Exptdata["Ra"][order], Exptdata["Dec"][order] = Ra, Dec
    else:
        Exptdata[f"RaOff_{i}"][order], Exptdata[f"DecOff_{i}"][order] = Ra, Dec


def RaDecerrCallback(result):
    print(result)


if __name__ == "__main__":
    DataPath = "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17_merged"
    SavePath = "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17_isgammacuted"
    SavePath2 = (
        "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17_isgammacuted_E_Ra_Dec"
    )
    SavePath3 = "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17_isgammacuted_E_Ra_Dec_new"
    mkdir(SavePath)
    mkdir(SavePath2)
    mkdir(SavePath3)

    # predictor_isgamma = TabularPredictor.load(
    #     "/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_Allsky_MC_5par_random_2/"
    # )

    # predictor_energy = TabularPredictor.load(
    #     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/log10Energy"
    # )
    # predictor_deltatheta = TabularPredictor.load(
    #     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltatheta"
    # )
    # predictor_deltaphi = TabularPredictor.load(
    #     "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltaphi"
    # )

    # # sumpfbins = np.logspace(1.6, 3, 8)
    # # P_value = np.array([0.36, 0.44, 0.55, 0.74, 0.91, 0.98, 0.99])

    datalist = list()
    for root, dirs, files in os.walk(SavePath2):
        for name in files:
            filename = os.path.join(root, name)
            datalist.append(filename)

    for filename in datalist:
        print(filename)
        # savefilename = getsavefilename(SavePath, filename)
        savefilename2 = getsavefilename(SavePath3, filename)
        if not os.path.isfile(savefilename2):
            Exptdata = np.load(filename)
            Exptdata = {key: Exptdata[key] for key in Exptdata}
            # Exptdata["P_value"] = np.zeros_like(Exptdata["summd"])
            # Exptdata["P_value"][Exptdata["sumpf"] < sumpfbins[0]] = P_value[0]
            # Exptdata["P_value"][Exptdata["sumpf"] > sumpfbins[-1]] = P_value[-1]
            # for i in range(len(P_value)):
            #     Exptdata["P_value"][
            #         (Exptdata["sumpf"] > sumpfbins[i])
            #         & (Exptdata["sumpf"] < sumpfbins[i + 1])
            #     ] = P_value[i]
            # Exptdata["isgamma"] = predictor_isgamma.predict_proba(pd.DataFrame(Exptdata))[
            #     1
            # ].to_numpy()
            # # np.savez(savefilename, **Exptdata)
            # cuted = np.where(Exptdata["isgamma"] > 0.01)
            # Exptdata = {key: Exptdata[key][cuted] for key in Exptdata.keys()}

            # np.savez(savefilename, **Exptdata)
            # Exptdata_df = pd.DataFrame(Exptdata)
            # Exptdata_df["sumpf"] = np.log10(Exptdata_df["sumpf"])
            # print("energy")
            # Exptdata["energy"] = 10 ** predictor_energy.predict(Exptdata_df).to_numpy()
            # print("newtheta")
            # Exptdata["newtheta"] = (
            #     Exptdata["theta"] - predictor_deltatheta.predict(Exptdata_df).to_numpy()
            # )
            # print("newphi")
            # Exptdata["newphi"] = (
            #     Exptdata["phi"] - predictor_deltaphi.predict(Exptdata_df).to_numpy()
            # )
            Exptdata["newtheta"][Exptdata["newtheta"] < 0] = 0.01
            shared_array_base = multiprocessing.Array(
                ctypes.c_double, 3 * len(Exptdata["newtheta"])
            )
            Exptdata["Ra"] = np.zeros_like(Exptdata["newtheta"])
            Exptdata["Dec"] = np.zeros_like(Exptdata["newtheta"])
            for i in range(20):
                Exptdata[f"RaOff_{i}"] = np.zeros_like(Exptdata["newtheta"])
                Exptdata[f"DecOff_{i}"] = np.zeros_like(Exptdata["newtheta"])

            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            shared_array = shared_array.reshape(3, len(Exptdata["newtheta"]))
            shared_array[0, :] = Exptdata["newtheta"][:]
            shared_array[1, :] = Exptdata["newphi"][:]
            shared_array[2, :] = Exptdata["mjd"][:]
            order = np.array_split(
                np.arange(len(Exptdata["newtheta"])),
                math.ceil(len(Exptdata["newtheta"]) / 1e5),
            )
            pool = multiprocessing.Pool(processes=21)
            for i in range(21):
                for order_tmp in order:
                    pool.apply_async(
                        getRaDecOff,
                        args=(i, shared_array[:, order_tmp], order_tmp),
                        callback=RaDecCallback,
                        error_callback=RaDecerrCallback,
                    )
            pool.close()
            pool.join()
            np.savez(savefilename2, **Exptdata)
