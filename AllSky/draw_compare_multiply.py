#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/Data/ZY/zy/code_Refactor/draw_compare.py
# Project: /home/hky/Data/ZY/zy/code_Refactor
# Created Date: 2022-08-18 14:09:28
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-08-23 15:16:04
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
###
import numpy as np
import matplotlib.pyplot as plt
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def draw_compare_multiply(
    datalist,
    legendlist,
    bins_begin=None,
    bins_end=None,
    bins_number=10,
    if_logx=0,
    if_logy=0,
    paraname=None,
    savepath=None,
    savename=None,
    xlabel=None,
):
    color = ["r", "b", 'g', "y", "b", "c"]
    if bins_begin is None:
        # bins_begin = np.min((np.min(MC), np.min(Expt)))
        bins_begin = np.min([np.min(x) for x in datalist])
        if if_logx:
            if bins_begin < 0:
                return
            bins_begin = np.log10(bins_begin)
    if bins_end is None:
        # bins_end = np.max((np.max(MC), np.max(Expt)))
        bins_end = np.max([np.max(x) for x in datalist])
        if if_logx:
            bins_end = np.log10(bins_end)
    if if_logx:
        bins = np.logspace(bins_begin, bins_end, num=bins_number)
    else:
        bins = np.linspace(bins_begin, bins_end, num=bins_number)
    if bins_begin == 0 and (if_logx == 0 or if_logy == 0):
        for data in datalist:
            data[data == 0] = 0.4
    bin_centres = (bins[:-1] + bins[1:]) / 2.0
    f, (a0, a1) = plt.subplots(
        2,
        1,
        figsize=(6, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for i, data in enumerate(datalist):
        counts_data, _ = np.histogram(
            data,
            bins=bins,
        )
        if i == 0:
            counts_data_Expt = counts_data
        # print(bin_centres)
        a0.errorbar(
            bin_centres,
            counts_data / np.sum(counts_data),
            yerr=np.sqrt(counts_data) / np.sum(counts_data),
            fmt="o",
            c=color[i],
            label=legendlist[i],
        )
        a1.errorbar(
            bin_centres,
            counts_data / np.sum(counts_data) /
            (counts_data_Expt / np.sum(counts_data_Expt)),
            yerr=np.sqrt(counts_data)
            / np.sum(counts_data)
            / (counts_data_Expt / np.sum(counts_data_Expt)),
            fmt="o",
            c=color[i],
            label=legendlist[i],
        )

    a1.set_ylim(0, 2)
    if if_logx:
        a1.set_xlim(10**bins_begin, 10**bins_end)
        a0.set_xlim(10**bins_begin, 10**bins_end)
    else:
        a1.set_xlim(bins_begin, bins_end)
        a0.set_xlim(bins_begin, bins_end)
    a0.legend(loc="best")
    if if_logx:
        a0.set_xscale("log")
        a1.set_xscale("log")
    a0.set_ylabel("dN/N")
    a1.set_ylabel(f"data/{legendlist[0]}")
    if xlabel is not None:
        a1.set_xlabel(xlabel)
    if if_logy:
        a0.set_yscale("log")
    if paraname is not None:
        a0.set_title(paraname)
    print(savename)
    # print(bins_begin, bins_end)
    if savepath is not None:
        mkdir(savepath)
        if savename is not None:
            plt.savefig(f"{savepath}/{savename}.png")
        else:
            plt.savefig(f"{savepath}/{paraname}.png")
        plt.close()
    else:
        plt.show()
