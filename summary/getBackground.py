import healpy as hp
import numpy as np
from tqdm import tqdm
import os
import sys


number = sys.argv[1]
print(number)
NSIDE = 2**9
NPIX = hp.nside2npix(NSIDE)
isgamma_bin = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
Energybin = np.logspace(0.6, 4, 18)
Energycenter = 10 ** ((np.log10(Energybin[1:]) + np.log10(Energybin[:-1])) / 2)
savePath = "/home2/hky/github/Gamma_Energy/summary/Data/splitMJD"
knowsource = np.load("/home2/hky/github/Gamma_Energy/summary/knownsource.npy")
Acceptance = np.load(os.path.join(savePath, "Acceptance.npy"))

Exptdata = np.load("/home2/hky/github/Gamma_Energy/summary/Exptdata_with.npz")

mjdcenter_sidereal = np.load(os.path.join(savePath, f"mjd_sidereal{number}.npy"))
mjdbegin = np.load(os.path.join(savePath, f"mjdbegin{number}.npy"))
mjdend = np.load(os.path.join(savePath, f"mjdend{number}.npy"))

Exptdataneed = np.where(
    (Exptdata["mjd"] >= np.min(mjdbegin)) & (Exptdata["mjd"] <= np.max(mjdend))
)
Exptdata = {key: Exptdata[key][Exptdataneed] for key in Exptdata}

pix_need = np.arange(NPIX)[np.where(np.sum(Acceptance, axis=0) != 0)]
Acceptance_need = Acceptance[:, pix_need]
theta, phi = hp.pix2ang(NSIDE, pix_need)

Background = np.zeros([len(isgamma_bin), len(Energybin) - 1, NPIX])
for mjd_i in range(len(mjdcenter_sidereal)):
    Background_pix = hp.ang2pix(
        NSIDE, theta, np.deg2rad(mjdcenter_sidereal[mjd_i]) - phi
    )
    rate = 1 / (
        1
        - np.sum(
            Acceptance[
                :,
                np.where(knowsource[Background_pix] == 1)[0],
            ],
            axis=(1),
        )
    )
    for i, isgammamin in enumerate(isgamma_bin):
        need = np.where(
            (Exptdata["mjd"] > mjdbegin[mjd_i])
            & (Exptdata["mjd"] < mjdend[mjd_i])
            & (Exptdata["ismasked"] == 0)
            & (Exptdata["isgamma"] > isgammamin)
        )
        N_background, _ = np.histogram(Exptdata["energy"][need], bins=Energybin)
        N_background = N_background * rate[i]
        for j in range(len(Energycenter)):
            if N_background[j] != 0:
                np.add.at(
                    Background[i, j],
                    Background_pix,
                    N_background[j] * Acceptance_need[i],
                )
np.save(os.path.join(savePath, f"Background{number}.npy"), Background)
