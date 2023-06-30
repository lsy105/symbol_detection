import numpy as np
from RC import RC

if __name__ == '__main__':
    silent = 1
    method = 'INV+RLS'
    N_total_frame = 17
    # N_total_frame = 94
    N_sync_frame = 4
    SNR = 10
    delay = 15
    window_size = 64
    N_reservoir = 64
    packet_num = 10

    ber_his = []
    dfe_ber_his = []
    LS_ber_his = []
    for i in range(packet_num):
        rc = RC(silent, method, N_total_frame, N_sync_frame, SNR, delay, window_size, N_reservoir)
        rc.i_packet = i
        ber, dfe_ber, LS_ber = rc.run()
        ber_his.append(ber)
        dfe_ber_his.append(dfe_ber)
        LS_ber_his.append(LS_ber)
        print('iteration', i, 'ber ', ber, 'dfe_ber', dfe_ber, 'LS_ber', LS_ber)
    print ('average_ber,', np.mean(ber_his), 'avg dfe ber,', np.mean(dfe_ber_his), 'avg LS ber,', np.mean(LS_ber_his))





