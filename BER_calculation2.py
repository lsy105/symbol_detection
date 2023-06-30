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
    FPGA_yout_complex = []
    for i in range(int(len(FPGA_yout) / 2)):
        FPGA_yout_complex.append(FPGA_yout[2 * i] + FPGA_yout[2 * i + 1] * 1j)
    # print(FPGA_yout_complex)

    rx_row = len(FPGA_yout) / 160
    FPGA_rx_time_cp = np.zeros([int(rx_row), 80], dtype='complex_')
    for i in range(int(rx_row)):
        for j in range(80):
            FPGA_rx_time_cp[i][j] = FPGA_yout_complex[i * 80 + j]
    return FPGA_rx_time_cp

def dat2rxtime(dat_name):
    # FPGA_yout = np.fromfile('fpga/y_out_hw.dat', sep='\n')
    FPGA_yout = np.fromfile(dat_name, sep='\n')
    rx_row = len(FPGA_yout) / 2
    FPGA_rx_time_cp = np.zeros([int(rx_row), 2])
    for i in range(int(rx_row)):
        for j in range(2):
            FPGA_rx_time_cp[i][j] = FPGA_yout[i * 2 + j]
    #print(FPGA_rx_time_cp)
    return FPGA_rx_time_cp


if __name__ == '__main__':
    silent = True
    method = 'INV+RLS'
    # N_total_frame = 17
    N_total_frame = 94  #94
    N_sync_frame = 4
    # SNR_list = np.arange(1,20,2)
    SNR_list = [1000]
    hw_ber = []
    sw_ber = []

    #select different rx_data adn tx_data for BER calculation
    folder_name = 'HW_results/S2'
    wifi_rx = np.load('data/S2/wifi_rx_time_cp.npy')
    tx_freq = np.load('data/S2/wifi_tx_freq.npy')
    #print(wifi_rx)

    #Dataset selection
    #folder_name = 'HW_results/S2/'  # LOS_Near:S2, LOS_Far:S3, NLOS:S1
    if folder_name == 'HW_results/S1':  # NLOS
        delay = 0
        packet_num = 21
    elif folder_name == 'HW_results/S2':  # LOS_Near
        delay = 1
        packet_num = 27  # correct
    elif folder_name == 'HW_results/S3':  # LOS_Far
        delay = 1
        packet_num = 22  # 23
    else:
        print("Undefined Dataset")
        exit(1)




    #delay = 1
    #packet_num = 22  # for testing

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

            outfile_hw = folder_name + '/' + str(i) + '/y_out_hw.dat'
            outfile_sw = folder_name + '/' + str(i) + '/predict_tosave.dat'
            FPGA_time = dat2rxtime(outfile_hw)
            python_time = dat2rxtime(outfile_sw)
            #FPGA_time = dat2rxtime('data_outputs/y_out_hw.dat')
            #tx_freq = np.load('data/S3/wifi_tx_freq.npy')
            #print('tx_freq:', tx_freq)
            #print('FPGA_time:', FPGA_time.shape, FPGA_time)
            FPGA_freq = rc.esn_output_to_block_f(FPGA_time, rc.N_data_frame, cp_removed=False, remove_delay=False)
            python_freq = rc.esn_output_to_block_f(python_time, rc.N_data_frame, cp_removed=False, remove_delay=False)
            #print('in1:', FPGA_freq[1:, :])
            #print('in2:', tx_freq[N_sync_frame + 1:, :])
            ber_ESN_hw = rc.ber(FPGA_freq[1:, :], tx_freq[N_sync_frame + 1:, :])
            ber_ESN_sw = rc.ber(python_freq[1:, :], tx_freq[N_sync_frame + 1:, :])
            hw_ber.append(ber_ESN_hw)
            sw_ber.append(ber_ESN_sw)
            print('packet_num: ' , i, ':   ber_ESN_hw: ', ber_ESN_hw, '   ber_ESN_sw: ', ber_ESN_sw)
            #rc.plot_constellation(FPGA_freq, 'ESN_data', pilot_removed=False)
            # with open ('data/check/FPGA_freq.txt','w+') as f1, open('data/check/tx_freq.txt','w+') as f2:
            #     for i in range(len(FPGA_freq)):
            #         f1.write(str(FPGA_freq[i]))
            #     for j in range(len(tx_freq)):
            #         f2.write(str(tx_freq[j]))
    average_hw_ber = sum(hw_ber) / len(hw_ber)
    print('average_hw_ber: ', average_hw_ber)
    average_sw_ber = sum(sw_ber) / len(sw_ber)
    print('average_sw_ber: ', average_sw_ber)
    # average_hw_ber_rmv = sum(hw_ber[4:]) / len(hw_ber[3:])
    # print('average_hw_ber_rmv: ', average_hw_ber_rmv)
    x = np.linspace(0, len(hw_ber) - 1, len(hw_ber))
    plt.figure(figsize=(8, 4))
    plt.plot(x, hw_ber, color='red', linewidth=2)
    plt.plot(x, sw_ber, color='blue', linewidth=2)
    plt.plot(x, np.array(hw_ber)-np.array(sw_ber), color='black', linewidth=2)
    plt.title('ber')
    plt.show()
