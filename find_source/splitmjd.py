import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import os


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


NSIDE = 2**10
NPIX = hp.nside2npix(NSIDE)
isgammacut = 0.8
savePath = f"/home2/hky/github/Gamma_Energy/find_source/data/isgammacut_{isgammacut}"
mjdcenter_sidereal = np.load(os.path.join(savePath, "mjd_sidereal.npy"))
mjdbins_begin = np.load(os.path.join(savePath, "mjdbegin.npy"))
mjdbins_end = np.load(os.path.join(savePath, "mjdend.npy"))

savePath = os.path.join(savePath, "splitmjd")

mkdirs(savePath)
split = np.array_split(np.arange(len(mjdbins_begin)), 40)
for i, j in enumerate(split):
    np.save(os.path.join(savePath, f"mjdbegin{i}.npy"), mjdbins_begin[j])
    np.save(os.path.join(savePath, f"mjdend{i}.npy"), mjdbins_end[j])
    np.save(os.path.join(savePath, f"mjd_sidereal{i}.npy"), mjdcenter_sidereal[j])
