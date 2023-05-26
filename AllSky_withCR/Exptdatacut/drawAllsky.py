import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings("ignore")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def top_hat(b, radius):
    return np.where(abs(b) <= radius, 1, 0)


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


def LIMA(alpha, Non, Noff):
    sig = np.sqrt(2) * np.sqrt(
        Non * np.log((1 + alpha) / alpha * (Non / (Non + Noff)))
        + Noff * np.log((1 + alpha) * Noff / (Non + Noff))
    )
    if type(sig) is np.ndarray:
        sig[np.where((Non - Noff * alpha) < 0)] = -sig[
            np.where((Non - Noff * alpha) < 0)
        ]
        sig[np.isnan(sig)] = 0

    else:
        if (Non - Noff * alpha) < 0:
            sig = -sig
        if np.isnan(sig):
            sig = 0
    return sig


NSIDE = 2**10
NPIX = hp.nside2npix(NSIDE)
reso = 1
drawdeg = 3
binsnumber = int(drawdeg * 60 / reso * 2)
Energybin = np.logspace(0.4, 3, 14)
Energybincenter = (Energybin[1:] + Energybin[:-1]) / 2
tst_map = np.zeros(NPIX)
pix_tst = hp.ang2pix(NSIDE, np.pi / 2, 0)
tst_map[pix_tst] = 100
b = np.linspace(0, np.pi, 100000)


DataPath = (
    "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17_isgammacuted_E_Ra_Dec_new"
)
Exptdata = dict()
for root, dirs, files in os.walk(DataPath):
    for name in files:
        Exptdata_tmp = np.load(os.path.join(root, name))
        Exptcut = np.where(Exptdata_tmp["isgamma"] > 0.2)
        for key in Exptdata_tmp:
            if key not in Exptdata.keys():
                Exptdata[key] = list()
            Exptdata[key].append(Exptdata_tmp[key][Exptcut])
for key in Exptdata.keys():
    Exptdata[key] = np.concatenate(Exptdata[key])
Exptdata["sumpfcut"] = np.zeros_like(Exptdata["sumpf"])


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


def getsigma_Allsky(cut, sumpfbins, Exptdata):
    Exptdata["sumpfcut"][Exptdata["sumpf"] < sumpfbins[0]] = 1
    Exptdata["sumpfcut"][Exptdata["sumpf"] > sumpfbins[-1]] = 0.99
    for i in range(len(cut)):
        Exptdata["sumpfcut"][
            (Exptdata["sumpf"] > sumpfbins[i]) & (Exptdata["sumpf"] < sumpfbins[i + 1])
        ] = cut[i]
    hpmap_All = np.zeros(NPIX)
    hpmap_Background = np.zeros(NPIX)
    need = np.where(Exptdata["isgamma"] > Exptdata["sumpfcut"])
    np.add.at(
        hpmap_All,
        hp.ang2pix(
            NSIDE,
            Exptdata["Ra"][need],
            Exptdata["Dec"][need],
            lonlat=True,
        ),
        1,
    )
    for i in range(20):
        np.add.at(
            hpmap_Background,
            hp.ang2pix(
                NSIDE,
                Exptdata[f"RaOff_{i}"][need],
                Exptdata[f"DecOff_{i}"][need],
                lonlat=True,
            ),
            1,
        )
    return hpmap_All, hpmap_Background


# pixlist = list()
sumpfbins = np.logspace(2, 4.8, 16)
sigmalist = list()
for cut0 in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
    cut = np.full(len(sumpfbins) - 1, cut0)
    hpmap_All, hpmap_Background = getsigma_Allsky(cut, sumpfbins, Exptdata)
    for smoothangle in [
        0.3,
        # 0.5,
        # 0.8,
        # 1,
        # 1.5,2
    ]:
        print(cut0, smoothangle)
        savepath = f"/home2/hky/github/Gamma_Energy/AllSky_withCR/Exptdatacut/fig/gammacut_{cut0:.2f}/smoothed{smoothangle:.1f}/"
        mkdir(savepath)
        bw = top_hat(b, np.radians(smoothangle))
        beam = hp.sphtfunc.beam2bl(bw, b, NSIDE * 3)
        tst_map_smoothed = hp.smoothing(tst_map, beam_window=beam)

        hpmap_All_smoothed = (
            hp.smoothing(hpmap_All, beam_window=beam)
            * tst_map[pix_tst]
            / np.mean(
                tst_map_smoothed[
                    hp.query_disc(
                        NSIDE, hp.pix2vec(NSIDE, pix_tst), np.radians(smoothangle)
                    )
                ]
            )
        )
        hpmap_Background_smoothed = (
            hp.smoothing(hpmap_Background, beam_window=beam)
            * tst_map[pix_tst]
            / np.mean(
                tst_map_smoothed[
                    hp.query_disc(
                        NSIDE, hp.pix2vec(NSIDE, pix_tst), np.radians(smoothangle)
                    )
                ]
            )
        )
        sigma = LIMA(0.05, hpmap_All_smoothed, hpmap_Background_smoothed)
        plt.hist(sigma[hpmap_All != 0], density=True, bins=100)
        x = np.linspace(-5, 5, 100)
        y = 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)
        plt.plot(x, y)
        plt.yscale("log")
        plt.savefig(os.path.join(savepath, "sigmahist.png"))
        plt.close()
        sigmalist.append(sigma)
sigmalist = np.array(sigmalist)
sigma_max = np.zeros_like(sigma)
for i in range(NPIX):
    sigma_max[i] = np.max(sigmalist[:, i])
plt.hist(sigma_max[hpmap_All != 0], density=True, bins=100)
x = np.linspace(-5, 5, 100)
y = 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)
plt.plot(x, y)
plt.yscale("log")
plt.savefig(
    os.path.join(
        "/home2/hky/github/Gamma_Energy/AllSky_withCR/Exptdatacut/fig/", "sigmahist.png"
    )
)
plt.close()

#         pixneed = np.where(sigma > 4.3)[0]
#         X = np.rad2deg(hp.pix2ang(NSIDE, pixneed)).T
#         distance = similarity(X)
#         model = DBSCAN(eps=1, min_samples=3, metric="precomputed")
#         yhat = model.fit_predict(distance)
#         sourcelist = np.unique(yhat)
#         for source_number in sourcelist:
#             if source_number == -1:
#                 continue
#             pix = pixneed[yhat == source_number][
#                 np.argmax(sigma[pixneed][yhat == source_number])
#             ]
#             pixlist.append(pix)
#             sigmamax = np.max(sigma[pixneed][yhat == source_number])
#             Ra, Dec = hp.pix2ang(NSIDE, pix, lonlat=True)
#             sigma_tmp = hp.gnomview(
#                 sigma,
#                 rot=[Ra, Dec],
#                 xsize=drawdeg * 60 / reso * 2,
#                 reso=reso,
#                 return_projected_map=True,
#                 no_plot=True,
#             )
#             sigma_inverse = np.zeros_like(sigma_tmp)
#             for i in range(sigma_tmp.shape[0]):
#                 sigma_inverse[:, i] = sigma_tmp[:, sigma_tmp.shape[0] - 1 - i]
#             fig, ax = plt.subplots(figsize=(16, 9))
#             c = ax.pcolormesh(
#                 np.linspace(
#                     Ra - drawdeg / np.cos(np.deg2rad(Dec)),
#                     Ra + drawdeg / np.cos(np.deg2rad(Dec)),
#                     binsnumber,
#                 ),
#                 np.linspace(
#                     Dec - drawdeg,
#                     Dec + drawdeg,
#                     binsnumber,
#                 ),
#                 sigma_inverse,
#                 cmap="plasma",
#                 vmin=0,
#             )
#             Ra_min = Ra - drawdeg / np.cos(np.deg2rad(Dec))
#             Ra_max = Ra + drawdeg / np.cos(np.deg2rad(Dec))
#             Dec_min = Dec - drawdeg
#             Dec_max = Dec + drawdeg
#             flag = 0
#             for Tname, Ttype, Ra2, Dec2 in zip(TeVname, TeVtype, Ra_TeVcat, Dec_TeVcat):
#                 if Ra_max > Ra2 > Ra_min and Dec_max > Dec2 > Dec_min:
#                     flag = 1
#                     ax.scatter(
#                         Ra2,
#                         Dec2,
#                         # c="r",
#                         # marker="x",
#                         label=f"{Tname}({Ttype})",
#                     )
#             fig.colorbar(c, orientation="vertical")
#             ax.set_xlim(
#                 Ra - drawdeg / np.cos(np.deg2rad(Dec)),
#                 Ra + drawdeg / np.cos(np.deg2rad(Dec)),
#             )
#             ax.set_ylim(Dec - drawdeg, Dec + drawdeg)
#             ax.invert_xaxis()
#             plt.title(f"{sigmamax:.2f}$\sigma$ {smoothangle:.1f}deg smoothed")
#             if flag == 1:
#                 plt.legend(bbox_to_anchor=(1.13, 0), loc="lower left")
#             plt.savefig(
#                 os.path.join(savepath, f"{Dec:.2f}_{Ra:.2f}_{sigmamax:.2f}.png")
#             )
#             plt.close()

#             need = np.where(
#                 (Exptdata["isgamma"] > Exptdata["sumpfcut"])
#                 & (np.abs(Exptdata["Ra"] - Ra) < 4 / np.cos(np.deg2rad(Dec)))
#                 & (np.abs(Exptdata["Dec"] - Dec) < 4)
#             )
#             distance = twoPointAngle(
#                 90 - Exptdata["Dec"][need], 90 - Dec, Exptdata["Ra"][need], Ra
#             )
#             Energy_tmp = Exptdata["energy"][need][distance < smoothangle]
#             On_hist, _ = np.histogram(Energy_tmp, Energybin)
#             Off_hist = np.zeros_like(On_hist)
#             for i in range(20):
#                 need = np.where(
#                     (Exptdata["isgamma"] > Exptdata["sumpfcut"])
#                     & (
#                         np.abs(Exptdata[f"RaOff_{i}"] - Ra)
#                         < 4 / np.cos(np.deg2rad(Dec))
#                     )
#                     & (np.abs(Exptdata[f"DecOff_{i}"] - Dec) < 4)
#                 )
#                 distance = twoPointAngle(
#                     90 - Exptdata[f"DecOff_{i}"][need],
#                     90 - Dec,
#                     Exptdata[f"RaOff_{i}"][need],
#                     Ra,
#                 )
#                 Energy_tmp = Exptdata["energy"][need][distance < smoothangle]
#                 Off_hist_tmp, _ = np.histogram(Energy_tmp, Energybin)
#                 Off_hist += Off_hist_tmp
#             plt.errorbar(
#                 Energybincenter, On_hist, yerr=np.sqrt(On_hist), fmt="o", label="On"
#             )
#             plt.errorbar(
#                 Energybincenter,
#                 Off_hist / 20,
#                 yerr=np.sqrt(Off_hist) / 20,
#                 fmt="o",
#                 label="Off",
#             )
#             plt.legend()
#             plt.xscale("log")
#             plt.xlabel("Energy(TeV)")
#             plt.ylabel("dN")
#             plt.savefig(
#                 os.path.join(savepath, f"{Dec:.2f}_{Ra:.2f}_{sigmamax:.2f}_OnOff.png")
#             )
#             plt.close()
# pixlist = np.array(pixlist)
# np.save("./pixlist.npy", pixlist)
# pixlist = np.load("./pixlist.npy")
# X = np.rad2deg(hp.pix2ang(NSIDE, pixlist)).T
# distance = similarity(X)
# model = DBSCAN(eps=0.5, min_samples=3, metric="precomputed")
# yhat = model.fit_predict(distance)
# sourcelist = np.unique(yhat)


# def drawsigma(sigma, Ra, Dec):
#     sigma_tmp = hp.gnomview(
#         sigma,
#         rot=[Ra, Dec],
#         xsize=drawdeg * 60 / reso * 2,
#         reso=reso,
#         return_projected_map=True,
#         no_plot=True,
#     )
#     sigma_inverse = np.zeros_like(sigma_tmp)
#     for i in range(sigma_tmp.shape[0]):
#         sigma_inverse[:, i] = sigma_tmp[:, sigma_tmp.shape[0] - 1 - i]
#     fig, ax = plt.subplots(figsize=(16, 9))
#     c = ax.pcolormesh(
#         np.linspace(
#             Ra - drawdeg / np.cos(np.deg2rad(Dec)),
#             Ra + drawdeg / np.cos(np.deg2rad(Dec)),
#             binsnumber,
#         ),
#         np.linspace(
#             Dec - drawdeg,
#             Dec + drawdeg,
#             binsnumber,
#         ),
#         sigma_inverse,
#         cmap="plasma",
#         vmin=0,
#     )
#     Ra_min = Ra - drawdeg / np.cos(np.deg2rad(Dec))
#     Ra_max = Ra + drawdeg / np.cos(np.deg2rad(Dec))
#     Dec_min = Dec - drawdeg
#     Dec_max = Dec + drawdeg
#     flag = 0
#     for Tname, Ttype, Ra2, Dec2 in zip(TeVname, TeVtype, Ra_TeVcat, Dec_TeVcat):
#         if Ra_max > Ra2 > Ra_min and Dec_max > Dec2 > Dec_min:
#             flag = 1
#             ax.scatter(
#                 Ra2,
#                 Dec2,
#                 # c="r",
#                 # marker="x",
#                 label=f"{Tname}({Ttype})",
#             )
#     fig.colorbar(c, orientation="vertical")
#     ax.set_xlim(
#         Ra - drawdeg / np.cos(np.deg2rad(Dec)),
#         Ra + drawdeg / np.cos(np.deg2rad(Dec)),
#     )
#     ax.set_ylim(Dec - drawdeg, Dec + drawdeg)
#     ax.invert_xaxis()
#     length = sigma_inverse.shape[0]
#     sigmamax = np.max(
#         sigma_inverse[
#             int(length / 3) : int(length * 2 / 3),
#             int(length / 3) : int(length * 2 / 3),
#         ]
#     )
#     plt.title(f"{sigmamax:.2f}$\sigma$ {smoothangle:.1f}deg smoothed")
#     if flag == 1:
#         plt.legend(bbox_to_anchor=(1.13, 0), loc="lower left")
#     plt.savefig(
#         os.path.join(
#             savepath,
#             f"{Dec:.2f}_{Ra:.2f}_{sigmamax:.2f}_{cut0:.2f}_{smoothangle:.1f}.png",
#         )
#     )
#     plt.close()
#     disc = hp.query_disc(NSIDE, hp.ang2vec(Ra, Dec, lonlat=True), np.deg2rad(1.5))
#     pix_maxsigma = disc[np.argmax(sigma[disc])]
#     Ra, Dec = hp.pix2ang(NSIDE, pix_maxsigma, lonlat=True)
#     need = np.where(
#         (Exptdata["isgamma"] > Exptdata["sumpfcut"])
#         & (np.abs(Exptdata["Ra"] - Ra) < 4 / np.cos(np.deg2rad(Dec)))
#         & (np.abs(Exptdata["Dec"] - Dec) < 4)
#     )
#     distance = twoPointAngle(
#         90 - Exptdata["Dec"][need], 90 - Dec, Exptdata["Ra"][need], Ra
#     )
#     Energy_tmp = Exptdata["energy"][need][distance < smoothangle]
#     On_hist, _ = np.histogram(Energy_tmp, Energybin)
#     Off_hist = np.zeros_like(On_hist)
#     for i in range(20):
#         need = np.where(
#             (Exptdata["isgamma"] > Exptdata["sumpfcut"])
#             & (np.abs(Exptdata[f"RaOff_{i}"] - Ra) < 4 / np.cos(np.deg2rad(Dec)))
#             & (np.abs(Exptdata[f"DecOff_{i}"] - Dec) < 4)
#         )
#         distance = twoPointAngle(
#             90 - Exptdata[f"DecOff_{i}"][need],
#             90 - Dec,
#             Exptdata[f"RaOff_{i}"][need],
#             Ra,
#         )
#         Energy_tmp = Exptdata["energy"][need][distance < smoothangle]
#         Off_hist_tmp, _ = np.histogram(Energy_tmp, Energybin)
#         Off_hist += Off_hist_tmp
#     plt.errorbar(Energybincenter, On_hist, yerr=np.sqrt(On_hist), fmt="o", label="On")
#     plt.errorbar(
#         Energybincenter,
#         Off_hist / 20,
#         yerr=np.sqrt(Off_hist) / 20,
#         fmt="o",
#         label="Off",
#     )
#     plt.legend()
#     plt.xscale("log")
#     plt.xlabel("Energy(TeV)")
#     plt.ylabel("dN")
#     plt.savefig(
#         os.path.join(
#             savepath,
#             f"{Dec:.2f}_{Ra:.2f}_{sigmamax:.2f}_{cut0:.2f}_{smoothangle:.1f}_OnOff.png",
#         )
#     )
#     plt.close()


# for cut0 in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
#     cut = np.full(len(sumpfbins) - 1, cut0)
#     hpmap_All, hpmap_Background = getsigma_Allsky(cut, sumpfbins, Exptdata)
#     for smoothangle in [0.3, 0.5, 0.8, 1, 1.5, 2]:
#         bw = top_hat(b, np.radians(smoothangle))
#         beam = hp.sphtfunc.beam2bl(bw, b, NSIDE * 3)
#         tst_map_smoothed = hp.smoothing(tst_map, beam_window=beam)

#         hpmap_All_smoothed = (
#             hp.smoothing(hpmap_All, beam_window=beam)
#             * tst_map[pix_tst]
#             / np.mean(
#                 tst_map_smoothed[
#                     hp.query_disc(
#                         NSIDE, hp.pix2vec(NSIDE, pix_tst), np.radians(smoothangle)
#                     )
#                 ]
#             )
#         )
#         hpmap_Background_smoothed = (
#             hp.smoothing(hpmap_Background, beam_window=beam)
#             * tst_map[pix_tst]
#             / np.mean(
#                 tst_map_smoothed[
#                     hp.query_disc(
#                         NSIDE, hp.pix2vec(NSIDE, pix_tst), np.radians(smoothangle)
#                     )
#                 ]
#             )
#         )
#         sigma = LIMA(0.05, hpmap_All_smoothed, hpmap_Background_smoothed)
#         savepath = f"/home2/hky/github/Gamma_Energy/AllSky_withCR/Exptdatacut/fig/eachsource/{-9.84:.2f}_{277.58:.2f}/"
#         mkdir(savepath)
#         drawsigma(sigma, 277.58, -9.84)
#         for source_number in sourcelist:
#             if source_number == -1:
#                 continue
#             pix = pixlist[yhat == source_number]
#             Ra, Dec = hp.pix2ang(NSIDE, pix, lonlat=True)
#             Ra = np.mean(Ra)
#             Dec = np.mean(Dec)
#             savepath = f"/home2/hky/github/Gamma_Energy/AllSky_withCR/Exptdatacut/fig/eachsource/{Dec:.2f}_{Ra:.2f}/"
#             mkdir(savepath)
#             drawsigma(sigma, Ra, Dec)
