#!/usr/bin/python

import numpy as np
from scipy.io import loadmat, savemat

wifi_tx_freq = np.load('wifi_tx_freq.npy')
savemat(file_name='wifi_tx_freq.mat', mdict={"wifi_tx_freq":wifi_tx_freq})


wifi_tx_time_cp = np.load('wifi_tx_time_cp.npy')
savemat(file_name='wifi_tx_time_cp.mat', mdict={"wifi_tx_time_cp":wifi_tx_time_cp})

wifi_rx_time_cp = np.load('wifi_rx_time_cp.npy')
savemat(file_name='wifi_rx_time_cp.mat', mdict={"wifi_rx_time_cp":wifi_rx_time_cp})




