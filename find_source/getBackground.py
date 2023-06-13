import healpy as hp
import numpy as np
import os
import sys


number = sys.argv[1]
NSIDE = 2**10
NPIX = hp.nside2npix(NSIDE)
isgammacut = 0.8
savePath = f"/home2/hky/github/Gamma_Energy/find_source/data/isgammacut_{isgammacut}"
Energybin = np.load(os.path.join(savePath, "Energybin.npy"))
Energycenter = np.load(os.path.join(savePath, "Energycenter.npy"))

knowsource = np.load("/home2/hky/github/Gamma_Energy/find_source/data/knowsource.npy")
Acceptance = np.load(os.path.join(savePath, "Acceptance.npy"))

Exptdata = np.load(os.path.join(savePath, "Exptdata.npz"))

savePath = os.path.join(savePath, "splitmjd")
mjdcenter_sidereal = np.load(os.path.join(savePath, f"mjd_sidereal{number}.npy"))
mjdbins_begin = np.load(os.path.join(savePath, f"mjdbegin{number}.npy"))
mjdbins_end = np.load(os.path.join(savePath, f"mjdend{number}.npy"))


Exptdataneed = np.where(
    (Exptdata["mjd"] >= np.min(mjdbins_begin))
    & (Exptdata["mjd"] <= np.max(mjdbins_end))
)
Exptdata = {key: Exptdata[key][Exptdataneed] for key in Exptdata}

pix_need = np.arange(NPIX)[np.where(Acceptance != 0)]
Acceptance_need = Acceptance[pix_need]
theta, phi = hp.pix2ang(NSIDE, pix_need)
Background = np.zeros([len(Energybin) - 1, NPIX])

for mjd_i in range(len(mjdcenter_sidereal)):
    Background_pix = hp.ang2pix(
        NSIDE, theta, np.deg2rad(mjdcenter_sidereal[mjd_i]) - phi
    )
    rate = 1 / (
        1
        - np.sum(
            Acceptance[np.where(knowsource[Background_pix] == 1),],
        )
    )
    need = np.where(
        (Exptdata["mjd"] > mjdbins_begin[mjd_i])
        & (Exptdata["mjd"] < mjdbins_end[mjd_i])
        & (Exptdata["ismasked"] == 0)
    )
    N_background, _ = np.histogram(Exptdata["energy"][need], bins=Energybin)
    N_background = N_background * rate
    for i in range(len(Energycenter)):
        np.add.at(
            Background[i],
            Background_pix,
            N_background[i] * Acceptance_need,
        )
np.save(os.path.join(savePath, f"Background{number}.npy"), Background)
