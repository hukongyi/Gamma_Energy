import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import uproot
from autogluon.tabular import TabularPredictor
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from getS50 import getS50

# import multiprocessing

from scipy.stats import poisson
from scipy.optimize import minimize

# from optimparallel import minimize_parallel
from tqdm.notebook import tqdm


def corrdinateToYBJ(Ra, Dec, mjd):
    """Transform J2000 to YBJ ALTAZ

    Args:
        Ra (np.array): Ra in J2000
        Dec (np.array): Dec in J2000
        mjd (np.array): mjd
    Returns:
        np.array: List of fo8ur np.array, first is theta with degree, second is phi with degree
    """
    YBJ_Location = EarthLocation(
        lat=30.102 * u.deg, lon=90.522 * u.deg, height=4300 * u.m
    )
    mjdtime = Time(mjd, format="mjd", location=YBJ_Location)
    newAltAzcoordiantes = SkyCoord(
        ra=Ra * u.deg,
        dec=Dec * u.deg,
        obstime=mjdtime,
        frame="icrs",
        location=YBJ_Location,
    )
    # LST = mjdtime.sidereal_time("apparent").degree
    alt = newAltAzcoordiantes.altaz.alt.degree
    az = newAltAzcoordiantes.altaz.az.degree
    return [
        90 - alt,
        az,
    ]


number = 100000
mjd = np.linspace(56710, 56715, number)
theta, phi = corrdinateToYBJ(0, 2.5, mjd)
time_rate = np.sum(theta < 60) / number

NSIDE = 2**10
NPIX = hp.nside2npix(NSIDE)

MCdata = uproot.open("/home2/hky/github/Gamma_Energy/MCdata/data_gamma_026/RESULT.root")

paraneed = [
    "nch",
    "inout",
    "theta",
    "phi",
    "sigma",
    "cx",
    "cy",
    "sumpf",
    "summd",
    "mr1",
    "ne",
    "age",
    "pritheta",
    "priphi",
    "prie",
]

MCdata = MCdata["asresult"].arrays(paraneed, library="np")
MCdata["S50"] = getS50(MCdata["ne"], MCdata["age"])
MCdatacuted = np.where(
    (MCdata["inout"] == 1)
    & (MCdata["sigma"] < 1)
    & (MCdata["nch"] >= 16)
    & (MCdata["theta"] < 60)
    & (MCdata["S50"] > 10**-1.2)
    & (MCdata["age"] > 0.31)
    & (MCdata["age"] < 1.3)
    & (MCdata["sumpf"] > 100)
    & ((MCdata["summd"] < 0.4) | (MCdata["summd"] < 1.2e-3 * MCdata["sumpf"] ** 1.6))
)

MCdata = {key: MCdata[key][MCdatacuted] for key in MCdata.keys()}
MCdata["pritheta"] = np.rad2deg(MCdata["pritheta"])
MCdata["priphi"] = np.rad2deg(MCdata["priphi"])
MCdata["priphi"] = 180 - MCdata["priphi"]
MCdata["priphi"][MCdata["priphi"] > 180] = (
    MCdata["priphi"][MCdata["priphi"] > 180] - 360
)
MCdata["prie"] = MCdata["prie"] / 1000

predictor = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/agmodel/identitfy_gamma_CR_Allsky_MC_5par_random_2"
)
predictor_energy = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/log10Energy"
)
predictor_deltatheta = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltatheta"
)
predictor_deltaphi = TabularPredictor.load(
    "/home2/hky/github/Gamma_Energy/AllSky/AutogluonModels/agModels_angle_ifcut=0/deltaphi"
)

MCdata_df = pd.DataFrame(MCdata)
MCdata["isgamma"] = np.zeros_like(MCdata["summd"])
MCdata["energy"] = np.zeros_like(MCdata["summd"])
MCdata["isgamma"] = predictor.predict_proba(MCdata_df)[1].to_numpy()
MCdata_df["sumpf"] = np.log10(MCdata_df["sumpf"])
MCdata["energy"] = 10 ** predictor_energy.predict(MCdata_df).to_numpy()
MCdata["newtheta"] = (
    MCdata["theta"] - predictor_deltatheta.predict(MCdata_df).to_numpy()
)
MCdata["newphi"] = MCdata["phi"] - predictor_deltaphi.predict(MCdata_df).to_numpy()

need = np.where(MCdata["isgamma"] > 0.8)
MCdata = {key: MCdata[key][need] for key in MCdata}

Energybin = np.load(
    "/home2/hky/github/Gamma_Energy/find_source/data/isgammacut_0.8/Energybin.npy"
)
priEnergybin = np.logspace(0.6, 4, 18)
PSF = np.zeros([len(Energybin) - 1, NPIX])

R_z = np.zeros([len(MCdata["pritheta"]), 3, 3])
R_y = np.zeros([len(MCdata["pritheta"]), 3, 3])

R_z[:, 2, 2] = 1
R_z[:, 0, 0] = np.cos(np.deg2rad(MCdata["priphi"]))
R_z[:, 0, 1] = np.sin(np.deg2rad(MCdata["priphi"]))
R_z[:, 1, 0] = -np.sin(np.deg2rad(MCdata["priphi"]))
R_z[:, 1, 1] = np.cos(np.deg2rad(MCdata["priphi"]))

R_y[:, 1, 1] = 1
R_y[:, 0, 0] = np.cos(np.deg2rad(MCdata["pritheta"]))
R_y[:, 0, 2] = -np.sin(np.deg2rad(MCdata["pritheta"]))
R_y[:, 2, 0] = np.sin(np.deg2rad(MCdata["pritheta"]))
R_y[:, 2, 2] = np.cos(np.deg2rad(MCdata["pritheta"]))

angvec = hp.ang2vec(np.deg2rad(MCdata["newtheta"]), np.deg2rad(MCdata["newphi"]))
for i in range(len(MCdata["pritheta"])):
    angvec[i] = np.matmul(R_z[i], angvec[i].T)
    angvec[i] = np.matmul(R_y[i], angvec[i].T)
    for j in range(len(Energybin) - 1):
        if (
            MCdata["energy"][i] > Energybin[j]
            and MCdata["energy"][i] < Energybin[j + 1]
        ):
            PSF[j, hp.vec2pix(NSIDE, *angvec[i])] += 1
            break

PSFl = list()
for i in range(PSF.shape[0]):
    PSFl.append(hp.map2alm(PSF[i] / np.sum(PSF[i]), mmax=0))
PSFl = np.array(PSFl)
PSFl = PSFl[2:-2, :]

hp.gnomview(np.sum(PSF, axis=0), rot=[0, 90], xsize=200, reso=1)
plt.title("PSF")

np.sum(PSF, axis=1)

Energybin = Energybin[2:-2]

PSF2 = np.zeros_like(PSF)
for i in range(PSF.shape[0]):
    PSF2[i] = hp.alm2map(hp.map2alm(PSF[i], mmax=0), NSIDE, lmax=3 * NSIDE - 1, mmax=0)
PSF2[np.isnan(PSF2)] = 0

hp.gnomview(np.sum(PSF2, axis=0), rot=[0, 90], xsize=200, reso=1)
plt.title("PSF_SH")

response = np.zeros([len(Energybin) - 1, len(priEnergybin) - 1])
for i in range(len(priEnergybin) - 1):
    response[:, i], _ = np.histogram(
        MCdata["energy"][
            (MCdata["prie"] > priEnergybin[i]) & (MCdata["prie"] < priEnergybin[i + 1])
        ],
        bins=Energybin,
    )

prie_hist, _ = np.histogram(MCdata["prie"], bins=priEnergybin)
energy_hist, _ = np.histogram(MCdata["energy"], bins=Energybin)
response = response / prie_hist

prie_hist

response[np.isnan(response)] = 0

pridata = uproot.open(
    "/home2/hky/github/Gamma_Energy/MCdata/data_gamma_026/priall.root"
)
pridata = pridata["tpri"].arrays(pridata["tpri"].keys(), library="np")

priorg_hist, _ = np.histogram(pridata["e"] / 1000, bins=priEnergybin)
eta = prie_hist / priorg_hist

del MCdata
del pridata


# Background = np.zeros([len(Energybin) - 1, NPIX])
# for i in range(40):
#     Background +=np.load(f"/home2/hky/github/Gamma_Energy/find_source/data/isgammacut_0.8/splitmjd/Background{i}.npy")
# np.save("/home2/hky/github/Gamma_Energy/find_source/data/isgammacut_0.8/splitmjd/Background.npy",Background)
Background = np.load(
    "/home2/hky/github/Gamma_Energy/find_source/data/isgammacut_0.8/splitmjd/Background.npy"
)

# DataPath = (
#     "/home2/hky/github/Gamma_Energy/Exptdata/ALLsky_23_05_17_isgammacuted_E_Ra_Dec_new"
# )
# Exptdata = dict()
# for root, dirs, files in os.walk(DataPath):
#     for name in files:
#         Exptdata_tmp = np.load(os.path.join(root, name))
#         Exptcut = np.where(Exptdata_tmp["isgamma"] > 0.6)
#         for key in Exptdata_tmp:
#             if key not in Exptdata.keys():
#                 Exptdata[key] = list()
#             Exptdata[key].append(Exptdata_tmp[key][Exptcut])
# for key in Exptdata.keys():
#     Exptdata[key] = np.concatenate(Exptdata[key])

# On  = np.zeros([len(Energybin) - 1, NPIX])
# for i in range(len(Energybin)-1):
#     need = np.where((Exptdata["energy"]>Energybin[i])&(Exptdata["energy"]<Energybin[i+1]))
#     np.add.at(On[i],hp.ang2pix(NSIDE,Exptdata["Ra"][need],Exptdata["Dec"][need],lonlat=True),1)
# np.save("/home2/hky/github/Gamma_Energy/J1857/direct_int_ag_background/On.npy",On)
# del Exptdata
On = np.load("/home2/hky/github/Gamma_Energy/find_source/data/isgammacut_0.8/All.npy")

Background = Background[2:-2, :]
On = On[2:-2]

# hp.mollview(hp.smoothing(np.sum(On - Background, axis=0), fwhm=np.deg2rad(0.5)))

# hp.gnomview(
#     hp.smoothing(np.sum(On - Background, axis=0), fwhm=np.deg2rad(0.5)),
#     rot=[284.5, 2.5],
#     xsize=200,
#     reso=1,
# )

# TeVdata = pd.read_table("/home2/hky/github/Gamma_Energy/AllSky_withCR/TeVcat.log")
# Ra_TeVcat = TeVdata["Ra"].to_numpy()
# Dec_TeVcat = TeVdata["Dec"].to_numpy()
# for i in range(len(Ra_TeVcat)):
#     Ra_TeVcat_tmp = Ra_TeVcat[i].split()
#     Dec_TeVcat_tmp = Dec_TeVcat[i].split()
#     Ra_TeVcat[i] = (
#         float(Ra_TeVcat_tmp[0]) / 24
#         + float(Ra_TeVcat_tmp[1]) / 24 / 60
#         + float(Ra_TeVcat_tmp[2]) / 24 / 60 / 60
#     ) * 360
#     Dec_TeVcat[i] = float(Dec_TeVcat_tmp[0])
#     delta_Dec_TeVcat = (
#         float(Dec_TeVcat_tmp[1]) / 60 + float(Dec_TeVcat_tmp[2]) / 60 / 60
#     )
#     Dec_TeVcat[i] += (-1) ** (Dec_TeVcat[i] < 0) * delta_Dec_TeVcat

# Ra_TeVcat = Ra_TeVcat.astype(np.float32)
# Dec_TeVcat = Dec_TeVcat.astype(np.float32)
# TeVname = TeVdata["Name"]
# TeVtype = TeVdata["Type"]

# # hp_map = hp.smoothing(np.sum(On-Background,axis=0),fwhm=np.radians(0.3))


# def drawgnomview(hp_map, Ra, Dec, reso=1, drawdeg=2):
#     binsnumber = int(drawdeg * 60 / reso * 2)
#     hp_map_tmp = hp.gnomview(
#         hp_map,
#         rot=[Ra, Dec],
#         xsize=drawdeg * 60 / reso * 2,
#         reso=reso,
#         return_projected_map=True,
#         no_plot=True,
#     )
#     hp_map_tmp_inv = np.zeros_like(hp_map_tmp)
#     for i in range(hp_map_tmp.shape[0]):
#         hp_map_tmp_inv[:, i] = hp_map_tmp[:, hp_map_tmp.shape[0] - 1 - i]
#     fig, ax = plt.subplots(figsize=(8, 5))
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
#         hp_map_tmp_inv,
#         cmap="plasma",
#         # vmin=3,
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
#     plt.legend()
#     plt.show()
#     return hp_map_tmp_inv


On_sum = np.sum(On, axis=0)
Background_sum = np.sum(Background, axis=0)


def getsigma(pix):
    PSF_All = np.sum(PSF, axis=0)
    PSF_All = hp.alm2map(hp.map2alm(PSF_All, mmax=0), NSIDE, lmax=3 * NSIDE - 1, mmax=0)
    PSF_All = PSF_All / np.sum(PSF_All)
    pix_need = hp.query_disc(NSIDE, hp.pix2vec(NSIDE, pix), np.deg2rad(3))
    Ra, Dec = hp.pix2ang(NSIDE, pix, lonlat=True)
    PSF_All_rotator = hp.Rotator(rot=[0, 90], deg=True)
    PSF_All_rotator2 = hp.Rotator(rot=[Ra, Dec], deg=True, inv=True)
    PSF_All = PSF_All_rotator2.rotate_map_pixel(
        PSF_All_rotator.rotate_map_pixel(PSF_All)
    )

    prob0 = poisson.pmf(On_sum[pix_need], Background_sum[pix_need])
    lnprob0 = np.sum(np.log(prob0))

    # print(lnprob0)
    def getprobA(A):
        # print(A)
        prob = poisson.pmf(
            On_sum[pix_need], Background_sum[pix_need] + A * PSF_All[pix_need]
        )
        return -np.sum(np.log(prob))

    A = 0
    res = minimize(getprobA, A, bounds=[(-1000, 1000)])
    # print(res)
    sig = np.sqrt(2 * (-res.fun - lnprob0))
    if res.x[0] < 0:
        sig = -sig

    return sig


sigma = np.zeros(NPIX)
for i in tqdm(
    np.arange(NPIX)[
        hp.query_disc(NSIDE, hp.ang2vec(284.5, 2.5, lonlat=True), np.deg2rad(3))
    ]
):
    sigma[i] = getsigma(i)

np.save("./sigma.npy", sigma)
