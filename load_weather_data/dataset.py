import numpy as np
import load_weather_data.utils as utils
from load_weather_data.arguments import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os

import scipy.io as sio

def get_data():
    # print(">>>In get_data")
    data = None
    var = DATA_FILE.split('.')[0].split('/')[-1]
    print(var)
    if DATA_FILE != "animal_data.mat":
        # print('RUNNING WEATHER DATA')
        data = utils.load_nc_data(DATA_FILE, variable=var)
    else:
        print(">>>Invalid data file")
        return None

    # normalize data between 0 and 1
    if NORMALIZE_Y:
        data = (data - data.min())/(data.max() - data.min())
    # print("data range: ", data.min(), data.max())
    data_mean = data.mean(0)
    # data = data - data_mean

    # print("data range: ", data.min(), data.max())

    if not RANDOM_ROI:
        data = data[:, LAT_MIN:LAT_MAX, LON_MIN:LON_MAX]
        data_mean = data_mean[LAT_MIN:LAT_MAX, LON_MIN:LON_MAX]
    else:
        lat_min = np.random.randint(0, data.shape[1]-ROI_SIZE) # 5) # data.shape[1]-ROI_SIZE*2)
        lat_max = lat_min + ROI_SIZE
        lon_min = np.random.randint(0, data.shape[2]-ROI_SIZE) # 20) #data.shape[2]-ROI_SIZE*2)
        lon_max = lon_min + ROI_SIZE
        data = data[:, lat_min:lat_max, lon_min:lon_max]
        data_mean = data_mean[lat_min:lat_max, lon_min:lon_max]

    return data, data_mean
    