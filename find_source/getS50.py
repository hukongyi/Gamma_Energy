import numpy as np
from scipy.special import gamma
from math import pi


def getS50(ne: np.ndarray, s: np.ndarray):
    """Get S50 from NKG function.

    Input:
        ne: Ne of Event.
        s: age of Event.

    Output:
        S50
    """
    rm = 130.
    r = 50.
    nr = ne / (rm * rm)
    gg1 = gamma(4.5 - s)
    gg2 = 2 * pi * gamma(s) * gamma(4.5 - 2 * s)
    rr1 = pow(r / rm, s - 2)
    rr2 = pow(1 + r / rm, s - 4.5)
    return nr * gg1 / gg2 * rr1 * rr2
