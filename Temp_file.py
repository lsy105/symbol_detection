import socket
import sys
import numpy as np
from RC_wifi import RC
import matplotlib.pyplot as plt
from scipy.io import savemat
from tanh import tanh


def dat2rx(dat_name):
    # FPGA_yout = np.fromfile('fpga/y_out_hw.dat', sep='\n')
    FPGA_yout = np.fromfile(dat_name, sep='\n')
    #print(FPGA_yout, len(FPGA_yout))
    #print(FPGA_yout[14399], type(FPGA_yout[14399]))

    FPGA_yout_complex = []
    for i in range(int(len(FPGA_yout) / 2)):
        FPGA_yout_complex.append(FPGA_yout[2 * i] + FPGA_yout[2 * i + 1] * 1j)
    # print(FPGA_yout_complex)

    rx_row = len(FPGA_yout) / 160
    #print(rx_row)
    FPGA_rx_time_cp = np.zeros([int(rx_row), 80], dtype='complex_')
    for i in range(int(rx_row)):
        for j in range(80):
            FPGA_rx_time_cp[i][j] = FPGA_yout_complex[i * 80 + j]
    # print(FPGA_rx_time_cp)
    return FPGA_rx_time_cp


if __name__ == '__main__':
    silent = True
    method = 'INV+RLS'
    # N_total_frame = 17
    N_total_frame = 90  #94
    N_sync_frame = 4
    # SNR_list = np.arange(1,20,2)
    SNR_list = [1000]

    # Dataset selection
    # folder_name = 'data/S2/'  # LOS_Near:S2, LOS_Far:S3, NLOS:S1
    # if folder_name == 'data/S1/':  # NLOS
    #     delay = 0
    #     packet_num = 21
    # elif folder_name == 'data/S2/':  # LOS_Near
    #     delay = 1
    #     packet_num = 27  # correct
    # elif folder_name == 'data/S3/':  # LOS_Far
    #     delay = 1
    #     packet_num = 22  # 23
    # else:
    #     print("Undefined Dataset")
    #     exit(1)


    #select different rx_data adn tx_data for BER calculation
    folder_name = 'data/FPGA_BER/'
    wifi_rx = np.load('data/S2/wifi_rx_time_cp.npy')
    print(wifi_rx)

    delay = 1
    packet_num = 1  # for testing

    window_size = 2
    N_reservoir = 16

    tanh_lut = tanh(
        input_bit=8,
        dx_bit=8,
        slope_fmt=(10, 10),
        intercept_fmt=(19, 19),
        max=8,
        better_lut=True,
        verbose=False,
        plot=False)

    for SNR in SNR_list:
        for i in range(packet_num):
            rc = RC(silent, method, N_total_frame, N_sync_frame, SNR, delay, window_size, i,
                    N_reservoir=16,
                    spectral_radius=0.2,
                    sparsity=0.4,
                    noise=1e-6,
                    lut_activation=False,  # True,
                    tanh_lut=tanh_lut,
                    input_scale=40,  # 50, # 25,
                    reservoir_input_scale=8,  # 4,  #5,
                    show_wout=False,
                    use_fpga=None,
                    sock=None,  # usock
                    addr=None)  # addr

            # ber, dfe_ber, LS_ber, comb_ber, sta_ber = rc.run()
            FPGA_rx_time_cp = dat2rx('fpga/y_out_hw.dat')
            print('in1:', FPGA_rx_time_cp.shape)
            print('in2:', rc.tx_time_cp[:-4].shape)
            ber = rc.train_and_test_inv_RLS(FPGA_rx_time_cp, rc.tx_time_cp[:-4])
            print('ber_new: ', ber)


