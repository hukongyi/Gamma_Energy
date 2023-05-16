import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
import pandas as pd
import warnings
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def LIMA(alpha, Non, Noff):
    sig = np.sqrt(2) * np.sqrt(
        Non * np.log((1 + alpha) / alpha * (Non / (Non + Noff)))
        + Noff * np.log((1 + alpha) * Noff / (Non + Noff))
    )
    if type(sig) is np.ndarray:
        sig[np.where((Non - Noff * alpha) < 0)] = -sig[
            np.where((Non - Noff * alpha) < 0)
        ]
    else:
        if (Non - Noff * alpha) < 0:
            sig = -sig
    sig[np.isnan(sig)] = 0
    return sig


def twoPointAngle(theta1, theta2, phi1, phi2):
    acosangle = np.sin(np.deg2rad(theta1)) * np.sin(np.deg2rad(theta2)) * np.cos(
        np.deg2rad(phi1 - phi2)
    ) + np.cos(np.deg2rad(theta1)) * np.cos(np.deg2rad(theta2))
    acosangle[acosangle > 1] = 1
    acosangle[acosangle < -1] = -1
    return np.rad2deg(np.arccos(acosangle))


def similarity(x):
    distance = np.zeros([x.shape[0], x.shape[0]])
    for i in range(x.shape[0]):
        # print(i)
        distance[i] = twoPointAngle(x[i, 0], x[:, 0], x[i, 1], x[:, 1])
    return distance


if __name__ == "__main__":
    # headers = [
    #     ">Energy(TeV)",
    #     "gammacut",
    #     "smoothed",
    #     "Ra",
    #     "Dec",
    #     "sigma",
    #     "source number",
    # ]
    Exptdata = np.load(
        "/home2/hky/github/Gamma_Energy/Exptdata/CrabCut_23_05_07_summdcut/cutedData_E_isgamma_RaDec.npz"
    )
    need = np.where(
        (Exptdata["energy"] > 10)
        # & (np.abs(Exptdata["cx"]) < 50)
        # & (np.abs(Exptdata["cy"]) < 50)
    )
    Exptdata = {key: Exptdata[key][need] for key in Exptdata}
    NSIDE = 2**10
    NPIX = hp.nside2npix(NSIDE)
    reso = 1
    drawdeg = 3
    binsnumber = int(drawdeg * 60 / reso * 2)
    TeVdata = pd.read_table("/home2/hky/github/Gamma_Energy/AllSky_withCR/TeVcat.log")
    Ra_TeVcat = TeVdata["Ra"].to_numpy()
    Dec_TeVcat = TeVdata["Dec"].to_numpy()
    for i in range(len(Ra_TeVcat)):
        Ra_TeVcat_tmp = Ra_TeVcat[i].split()
        Dec_TeVcat_tmp = Dec_TeVcat[i].split()
        Ra_TeVcat[i] = (
            float(Ra_TeVcat_tmp[0]) / 24
            + float(Ra_TeVcat_tmp[1]) / 24 / 60
            + float(Ra_TeVcat_tmp[2]) / 24 / 60 / 60
        ) * 360
        Dec_TeVcat[i] = float(Dec_TeVcat_tmp[0])
        delta_Dec_TeVcat = (
            float(Dec_TeVcat_tmp[1]) / 60 + float(Dec_TeVcat_tmp[2]) / 60 / 60
        )
        Dec_TeVcat[i] += (-1) ** (Dec_TeVcat[i] < 0) * delta_Dec_TeVcat

    Ra_TeVcat = Ra_TeVcat.astype(np.float32)
    Dec_TeVcat = Dec_TeVcat.astype(np.float32)
    TeVname = TeVdata["Name"]
    TeVtype = TeVdata["Type"]
    for para_num in [5]:
        # tablelist = list()
        for gammacut in [0.987]:
            for Energymin in [10, 25]:
                print(Energymin)
                On = np.zeros(NPIX)
                Off = np.zeros(NPIX)
                need = np.where(
                    (Exptdata["energy"] > Energymin)
                    & (Exptdata[f"isgamma_{para_num}_useMC"] > gammacut)
                )
                np.add.at(
                    On,
                    hp.ang2pix(
                        NSIDE, Exptdata["Ra"][need], Exptdata["Dec"][need], lonlat=True
                    ),
                    1,
                )
                for i in range(20):
                    np.add.at(
                        Off,
                        hp.ang2pix(
                            NSIDE,
                            Exptdata[f"RaOff_{i}"][need],
                            Exptdata[f"DecOff_{i}"][need],
                            lonlat=True,
                        ),
                        1,
                    )

                for smoothed in [
                    0.5,
                    0.8,
                    1,
                    1.5,
                ]:
                    print(smoothed)
                    fwhm = np.deg2rad(smoothed)
                    On_smoothed = hp.smoothing(On, fwhm=fwhm)
                    Off_smoothed = hp.smoothing(Off, fwhm=fwhm)

                    savepath = f"/home2/hky/github/Gamma_Energy/AllSky_withCR/fig/All_sky_new_{para_num}_useMC_withoutTeVCat/gammacut_{gammacut:.4f}/Over{Energymin}TeV/smoothed{smoothed:.1f}/"
                    mkdir(savepath)
                    sigma_hp = LIMA(0.05, On_smoothed, Off_smoothed)
                    std = np.std(sigma_hp[(On != 0)])
                    std = np.std(sigma_hp[(On != 0) & (np.abs(sigma_hp) < 5 * std)])
                    sigma_hp /= std
                    pixneed = np.where(sigma_hp > 4.5)[0]
                    print(len(pixneed))
                    if len(pixneed) > 0:
                        X = np.rad2deg(hp.pix2ang(NSIDE, pixneed)).T
                        distance = similarity(X)

                        model = DBSCAN(eps=1, min_samples=4, metric="precomputed")
                        yhat = model.fit_predict(distance)
                        del model
                        sourcelist = np.unique(yhat)[1:]
                        for source_number in sourcelist:
                            pix = pixneed[yhat == source_number][
                                np.argmax(sigma_hp[pixneed][yhat == source_number])
                            ]
                            sigmamax = np.max(sigma_hp[pixneed][yhat == source_number])
                            Ra, Dec = hp.pix2ang(NSIDE, pix, lonlat=True)
                            angle_distance = twoPointAngle(
                                90 - Dec, 90 - Dec_TeVcat, Ra, Ra_TeVcat
                            )
                            source_number = np.sum(angle_distance < 1)
                            if source_number >= 0:
                                sigma_tmp = hp.gnomview(
                                    sigma_hp,
                                    rot=[Ra, Dec],
                                    xsize=drawdeg * 60 / reso * 2,
                                    reso=reso,
                                    return_projected_map=True,
                                    no_plot=True,
                                )
                                sigma_inverse = np.zeros_like(sigma_tmp)
                                for i in range(sigma_tmp.shape[0]):
                                    sigma_inverse[:, i] = sigma_tmp[
                                        :, sigma_tmp.shape[0] - 1 - i
                                    ]
                                # tablelist.append(
                                #     [
                                #         Energymin,
                                #         gammacut,
                                #         smoothed,
                                #         Ra,
                                #         Dec,
                                #         sigmamax,
                                #         source_number,
                                #     ]
                                # )
                                fig, ax = plt.subplots(figsize=(16, 9))
                                c = ax.pcolormesh(
                                    np.linspace(
                                        Ra - drawdeg / np.cos(np.deg2rad(Dec)),
                                        Ra + drawdeg / np.cos(np.deg2rad(Dec)),
                                        binsnumber,
                                    ),
                                    np.linspace(
                                        Dec - drawdeg,
                                        Dec + drawdeg,
                                        binsnumber,
                                    ),
                                    sigma_inverse,
                                    cmap="plasma",
                                    vmin=0,
                                )
                                Ra_min = Ra - drawdeg / np.cos(np.deg2rad(Dec))
                                Ra_max = Ra + drawdeg / np.cos(np.deg2rad(Dec))
                                Dec_min = Dec - drawdeg
                                Dec_max = Dec + drawdeg
                                for Tname, Ttype, Ra2, Dec2 in zip(
                                    TeVname, TeVtype, Ra_TeVcat, Dec_TeVcat
                                ):
                                    if (
                                        Ra_max > Ra2 > Ra_min
                                        and Dec_max > Dec2 > Dec_min
                                    ):
                                        ax.scatter(
                                            Ra2,
                                            Dec2,
                                            # c="r",
                                            # marker="x",
                                            label=f"{Tname}({Ttype})",
                                        )
                                fig.colorbar(c, orientation="vertical")
                                ax.set_xlim(
                                    Ra - drawdeg / np.cos(np.deg2rad(Dec)),
                                    Ra + drawdeg / np.cos(np.deg2rad(Dec)),
                                )
                                ax.set_ylim(Dec - drawdeg, Dec + drawdeg)
                                ax.invert_xaxis()
                                plt.title(
                                    f"E>{Energymin} TeV {sigmamax:.2f}$\sigma$ {smoothed:.1f}deg smoothed"
                                )
                                plt.legend(bbox_to_anchor=(1.13, 0), loc="lower left")
                                plt.savefig(
                                    os.path.join(savepath, f"{Dec:.2f}_{Ra:.2f}_{sigmamax:.2f}.png")
                                )
                                plt.close()
        # print(tabulate(tablelist, headers=headers, tablefmt="grid", floatfmt=".2f"))
        # result = pd.DataFrame(tablelist, columns=headers)
        # result.to_csv(
        #     f"/home2/hky/github/Gamma_Energy/AllSky_withCR/fig/All_sky/result_{para_num}.csv",
        #     index=False,
        # )
