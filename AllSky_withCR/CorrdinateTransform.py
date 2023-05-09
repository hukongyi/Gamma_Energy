#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/PythonCode/point-source-python/CorrdinateTransform.py
# Project: /home/hky/PythonCode/point-source-python
# Created Date: 2022-03-08 17:02:01
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-12-14 12:15:34
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-12-14	K.Y.Hu		remove angle and LST and add corrdinateToYBJ
# 2022-11-22	K.Y.Hu		Add hour angle and LST
# 2022-03-14	K.Y.Hu		Modify docstring
# 2022-03-08	K.Y.Hu		Add introdction of corrdinateYBJ
# 2022-03-08	K.Y.Hu		Create function corrdinateYBJ
###

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time


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
    altaz = newAltAzcoordiantes.altaz
    alt = altaz.alt.degree
    az = altaz.az.degree
    return [
        90 - alt,
        az,
    ]


def corrdinateYBJ(theta, phi, mjd):
    """Transform YBJ ALTAZ to J2000

    Args:
        theta (np.array): zenith angle in location coordinates
        phi (np.array): Azimuth angle in location corrdinates
        mjd (np.array): mjd
    Returns:
        np.array: List of two np.array, first is ra with degree, second is dec with degree.
    """
    # start = time.time()
    # print("===========start========")
    YBJ_Location = EarthLocation(
        lat=30.102 * u.deg, lon=90.522 * u.deg, height=4300 * u.m
    )
    mjdtime = Time(mjd, format="mjd", location=YBJ_Location)
    # print("时间变换: ", time.time() - start)
    newAltAzcoordiantes = SkyCoord(
        alt=(90 - theta) * u.deg,
        az=phi * u.deg,
        obstime=mjdtime,
        frame="altaz",
        location=YBJ_Location,
    )
    # print("坐标变换: ", time.time() - start)
    RaDec = newAltAzcoordiantes.icrs
    Ra = RaDec.ra.degree
    Dec = RaDec.dec.degree
    # print("RaDec计算: ", time.time() - start)
    return [
        Ra,
        Dec,
    ]


def corrdinateYBJGalactic(theta, phi, mjd):
    """Transform YBJ ALTAZ to J2000

    Args:
        theta (np.array): zenith angle in location coordinates
        phi (np.array): Azimuth angle in location corrdinates
        mjd (np.array): mjd
    Returns:
        np.array: List of two np.array, first is ra with degree, second is dec with degree.
    """
    # start = time.time()
    # print("===========start========")
    YBJ_Location = EarthLocation(
        lat=30.102 * u.deg, lon=90.522 * u.deg, height=4300 * u.m
    )
    mjdtime = Time(mjd, format="mjd", location=YBJ_Location)
    newAltAzcoordiantes = SkyCoord(
        alt=(90 - theta) * u.deg,
        az=phi * u.deg,
        obstime=mjdtime,
        frame="altaz",
        location=YBJ_Location,
    )
    lb = newAltAzcoordiantes.galactic
    Galacticl = lb.l.degree
    Galacticb = lb.b.degree
    return [
        Galacticl,
        Galacticb,
    ]
