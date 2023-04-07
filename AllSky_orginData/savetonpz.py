#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /home/hky/github/Identification_CR/P00_readMCdata/savetonpz.py
# Project: /home/hky/github/Identification_CR/P00_readMCdata
# Created Date: 2022-06-15 15:47:18
# Author: Hu Kongyi
# Email:hukongyi@ihep.ac.cn
# -----
# Last Modified: 2022-06-17 16:31:55
# Modified By: Hu Kongyi
# -----
# HISTORY:
# Date      	By      	Comments
# ----------	--------	----------------------------------------------------
# 2022-06-17	K.Y.Hu		change save format
# 2022-06-17	K.Y.Hu		add indices in npz of each event
# 2022-06-17	K.Y.Hu		fix bug of exchange 2 line of numpy
# 2022-06-16	K.Y.Hu		finsh method
# 2022-06-16	K.Y.Hu		add save to numpy method
# 2022-06-15	K.Y.Hu		Create file
###
import struct
import numpy as np
import time
import datetime
import gzip
import os
import uproot
from tqdm import tqdm


class One_trigger(object):
    """get data for one event"""

    def __init__(self, buf: bytes, output_struct: struct.Struct):
        """init

        Args:
            buf (bytes): read data using binary
            output_struct (struct.Struct): data form with struct define by c++
        """
        output = output_struct.unpack_from(buf)
        self.pri_numpy = np.array(output[:11])
        # self.pri = {
        #     'e_num': output[0],
        #     'k': output[1],
        #     'id': output[2],
        #     'e': output[3],
        #     'theta': output[4],
        #     'phi': output[5],
        #     'ne': output[6],
        #     'core_x': output[7],
        #     'core_y': output[8],
        #     'sump': output[9],
        #     'n_hit': output[10],
        # }
        self.prtcl = np.array(output[11:1007])
        self.timing = np.array(output[1007:2003])  # 光束到平面的时间为零点
        self.photon = np.array(output[2003:]).reshape([4, 16, 4])
        switch_order = [12, 13, 14, 15, 8, 9, 10, 11]
        for i in range(8):
            self.switch_photon(i, switch_order[i])

    def switch_photon(self, i: int, j: int):
        """switch i,j because of some mistakes in MD order

        Args:
            i (int): first order for switch
            j (int): second order for switch
        """
        self.photon[:, [i, j], :] = self.photon[:, [j, i], :]


class Data(object):
    """Storing data for all MC event by sparse matrix
    can not using list(out of memory)
    """

    def __init__(self, MCnumber):
        # primary particle information
        self.pri = np.zeros([MCnumber, 11])
        # Tibet-III event number correspond Tibet
        # with 3 dimension,
        # number of event,
        # number of fire detector,
        # number of photons, time
        self.Tibet = np.zeros([MCnumber, 996, 2], dtype="float32")
        # with 4 element,
        # number of event,
        # number of MD,
        # number of pool,
        # number of PMT,
        self.MD = np.zeros([MCnumber, 4, 16, 4], dtype="float32")
        self.count = 0

    def addevent(self, trigger: One_trigger, MCdata, pbar):
        """add one event from trigger

        Args:
            trigger (One_trigger): read data for one event
        """
        if (
            trigger.pri_numpy[0] == MCdata["prie_num"][self.count]
            and trigger.pri_numpy[1] == MCdata["prik"][self.count]
        ):
            pbar.update(1)

            self.pri[self.count] = trigger.pri_numpy

            self.Tibet[self.count, :, 0] = trigger.prtcl
            self.Tibet[self.count, :, 1] = trigger.timing
            self.MD[self.count] = trigger.photon
            self.count += 1

    def save(self, savepath: str, savename: str, MCdata, compressed: bool = False):
        """save to savepath with npz

        Args:
            savepath (str): path to save
            compressed (bool): save with savez or savez_compressed
        """
        savedata = dict()
        savedata["prie_num"] = self.pri[:, 0].astype(int)
        savedata["priid"] = self.pri[:, 2].astype(int)
        savedata["prie"] = self.pri[:, 3]
        savedata["pritheta"] = self.pri[:, 4]
        savedata["priphi"] = self.pri[:, 5]
        savedata["prine"] = self.pri[:, 6].astype(int)
        savedata["pricx"] = self.pri[:, 7]
        savedata["pricy"] = self.pri[:, 8]
        savedata["Tibet"] = self.Tibet
        savedata["MD"] = self.MD
        for i in MCdata.keys():
            if i not in savedata.keys():
                savedata[i] = MCdata[i]
        if compressed:
            np.savez_compressed(os.path.join(savepath, savename + ".npz"), **savedata)
        else:
            np.savez(os.path.join(savepath, savename + ".npz"), **savedata)


def savetonpz(
    originpath: str,
    savepath: str,
    savename: str,
    MCnumber,
    MCdata,
    compressed: bool = False,
):
    """resave data to npz file

    Args:
        originpath (str): data path to read
        savepath (str): path to save
        savename (str): name to save
        MCnumber:envet number for MC after cut
        MCdata:MCdata
        compressed (bool): save with savez or savez_compressed

    Returns:
        str: messgae to send to wecom
    """
    start_time = time.time()
    output_struct = struct.Struct("3i3di3di1992d256i")

    data_numpy = Data(MCnumber)
    with tqdm(total=MCnumber, unit="event", unit_scale=True, mininterval=10) as pbar:
        with gzip.open(originpath, "rb") as data_file:
            while 1:
                buf = data_file.read(output_struct.size)
                if len(buf) != output_struct.size:
                    break
                trigger = One_trigger(buf, output_struct)
                data_numpy.addevent(trigger, MCdata, pbar)
                if data_numpy.count == MCnumber:
                    break
    data_numpy.save(savepath, savename, MCdata, compressed)
    # 15-20 min
    print(f"用时：{datetime.timedelta(seconds=time.time() - start_time)}")
    return f"save in {savepath} success!"


def getMCcutNumber(MCpath):
    file = uproot.open(MCpath)
    data = file["asresult"].arrays(
        [
            "nch",
            "inout",
            "theta",
            "phi",
            "sigma",
            "cx",
            "cy",
            "sumpf",
            "sumpd",
            "mr1",
            "summd",
            "ne",
            "age",
            "prie_num",
            "prik",
            "priid",
            "pritheta",
            "priphi",
            "prie",
            "prine",
            "pricx",
            "pricy",
        ],
        library="np",
    )
    cuted = np.where(data["inout"] == 1)
    data = {key: data[key][cuted] for key in data.keys()}
    return cuted[0].shape[0], data


if __name__ == "__main__":
    originpath = "/home2/chenxu/data/gamma_allsky.gz"
    MCpath = "/home2/chenxu/data/gamma_all.root"
    savepath = "/home2/hky/github/Gamma_Energy/AllSky_orginData/Data/"
    savename = "gamma_allsky"
    MCnumber, MCdata = getMCcutNumber(MCpath)
    savetonpz(originpath, savepath, savename, MCnumber, MCdata)
