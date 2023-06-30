import socket
import sys
import numpy as np
from RC_wifi import RC
import matplotlib.pyplot as plt
from scipy.io import savemat

from tanh import tanh

if __name__ == '__main__':
    silent = True
    method = 'RLS'  # RLS; INV; INV+RLS
    # N_total_frame = 17
    N_total_frame = 94
    N_sync_frame = 4
    # SNR_list = np.arange(1,20,2)
    SNR_list = [55]

    # Dataset selection
    folder_name = 'data/S2/'  # LOS_Near:S2, LOS_Far:S3, NLOS:S1
    output_folder = 'data_outputs/S2'

    if folder_name == 'data/S1/':  # NLOS
        delay = 0
        packet_num = 21
    elif folder_name == 'data/S2/':  # LOS_Near
        delay = 1
        packet_num = 27 # correct
    elif folder_name == 'data/S3/':  # LOS_Far
        delay = 1
        packet_num = 22 # 23c
    else:
        print("Undefined Dataset")
        exit(1)

    #packet_num = 22  # for testing 22
    debug = False

    # UDP send and receive sockets
    # addr = ("192.168.20.20", 8001)
    # usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # try:
    #     usock.bind(("", addr[1]))
    # except socket.error as msg:
    #     print(msg)
    #     sys.exit()



    window_size = 2
    N_reservoir = 16

    ber_record = []
    dfe_ber_record = []
    LS_ber_record = []
    comb_ber_record = []
    sta_ber_record = []
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
        ber_his = []
        dfe_ber_his = []
        LS_ber_his = []
        comb_ber_his = []
        sta_ber_his = []
        #for i in range(packet_num):
        for i in [11, 13, 14, 16, 17, 18, 19, 20]:
            rc = RC(silent, method, N_total_frame, N_sync_frame, SNR, delay, window_size, i,
                    N_reservoir=16,
                    spectral_radius=0.2,
                    sparsity=0.4,
                    noise=1e-6,
                    lut_activation=False,  # True,
                    tanh_lut=tanh_lut,
                    input_scale=25,  #40, #50, # 25,
                    reservoir_input_scale = 8,  #4,  #5,
                    show_wout=False,
                    output_folder= output_folder,
                    debug=debug,
                    use_fpga= None,
                    sock= None,  # usock
                    addr = None) # addr

            # rc.i_packet = i
            ber, dfe_ber, LS_ber, comb_ber, sta_ber = rc.run()
            ber_his.append(ber)
            dfe_ber_his.append(dfe_ber)
            LS_ber_his.append(LS_ber)
            comb_ber_his.append(comb_ber)
            sta_ber_his.append(sta_ber)
            print('iteration', i, 'ber ', ber, 'dfe_ber', dfe_ber, 'LS_ber',
                  LS_ber, 'comb_ber', comb_ber, 'sta_ber', sta_ber)
        print('SNR,', SNR, 'average_ber,', np.mean(ber_his), 'avg dfe ber,', np.mean(dfe_ber_his), 'avg LS ber,',
              np.mean(LS_ber_his), 'avg comb ber,', np.mean(comb_ber_his), 'avg sta ber,', np.mean(sta_ber_his))
        ber_record.append(np.mean(ber_his))
        #print('ber_his:', ber_his)
        dfe_ber_record.append(np.mean(dfe_ber_his))
        LS_ber_record.append(np.mean(LS_ber_his))
        comb_ber_record.append(np.mean(comb_ber_his))
        sta_ber_record.append(np.mean(sta_ber_his))

    '''
    savemat(file_name='fpga/test_rc_wifi_results.mat',
            mdict={"ber_record": ber_record, "dfe_ber_record": dfe_ber_record, "ls_ber_record": LS_ber_record,
                   "comb_ber_record": comb_ber_record, "sta_ber_record": sta_ber_record})
    
    plt.figure()
    plt.semilogy(SNR_list, ber_record, 'o-', label="ESN")
    plt.semilogy(SNR_list, dfe_ber_record, 'o-', label="DFE")
    plt.semilogy(SNR_list, LS_ber_record, 'o-', label="LS")
    plt.semilogy(SNR_list, comb_ber_record, 'o-', label="Comb")
    plt.semilogy(SNR_list, sta_ber_record, 'o-', label="STA")
    plt.title('BER')
    plt.ylabel('BER')
    plt.xlabel('SNR')
    plt.legend(loc=0)
    plt.grid(True, which="both", ls="-")
    
    plt.show()
    '''
