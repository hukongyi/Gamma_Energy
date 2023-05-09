from CorrdinateTransform import corrdinateYBJGalactic
import ctypes
import os
import numpy as np
import multiprocessing
import math
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split


Data = np.load(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Datawithe_Galactic_1000.npz"
)
Data = {key: Data[key] for key in Data}
Data["isgamma"] = np.where(np.abs(Data["b"] < 6), 1, 0)
train_index, test_index = train_test_split(
    range(len(Data["nch"])), test_size=0.4, random_state=42
)
trainData = {key: Data[key][train_index] for key in Data.keys()}
testData = {key: Data[key][test_index] for key in Data.keys()}
np.savez(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Datawithe_Galactic_1000_train.npz",
    **trainData
)
np.savez(
    "/home2/hky/github/Gamma_Energy/AllSky_withCR/Data/Datawithe_Galactic_1000_test.npz",
    **testData
)
