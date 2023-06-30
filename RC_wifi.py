import numpy as np
import matplotlib.pyplot as plt
import copy
from pyESN import ESN
from scipy import interpolate
# import tanh as tanh

class RC():
    def __init__(
            self,
            silent=True,
            method='INV+RLS',
            N_total_frame=17,
            N_sync_frame=4,
            SNR=100,
            delay=15,
            window_size=64,
            i=0,
            N_reservoir=32,
            # New parameters
            spectral_radius=0.2,
            sparsity=0.4,
            noise=1e-6,
            folder_name='data/S2/',
            input_scale=10,
            reservoir_input_scale = 1.0,
            lut_activation=False,
            tanh_lut=None,
            show_wout=False,
            use_fpga=False,
            sock=[],
            debug = False,  #2022
            savedat = False,  #2022
            output_folder = 'data_outputs/S2/',  #2022
            addr = ""):

        self.debug = debug
        self.savedat = savedat
        self.output_folder = output_folder

        self.silent = silent
        self.lut_activation = lut_activation
        self.tanh_lut = tanh_lut

        self.input_scale = input_scale # optimal for 1M and 200k dataset
        self.reservoir_input_scale = reservoir_input_scale
        # self.label_scale = 0.0001
        self.label_scale = 1

        self.method = method
        self.i_packet = i
        self.N_total_frame = N_total_frame
        self.N_sync_frame = N_sync_frame
        # self.N_data_frame = 1
        self.N_data_frame = self.N_total_frame - self.N_sync_frame
        self.cp_len = 16
        self.fft_size = 64
        self.SNR = SNR

        self.delay = delay
        self.window_size = window_size

        self.N_reservoir = N_reservoir
        self.spectral_radius = spectral_radius
        # self.spectral_radius = 0.05  #200k_10p_3 better in low SNR
        self.sparsity = sparsity
        self.noise = noise
        folder_name = folder_name
        self.show_wout = show_wout  # show the out weights?

        self.use_fpga = use_fpga
        self.sock = sock
        self.addr = addr

        self.rx_time_cp_all_raw = np.load(folder_name + '/wifi_rx_time_cp.npy')
        self.rx_time_cp_all = self.remove_zeros(self.rx_time_cp_all_raw)

        # self.carrier = np.array(range(-26, -21) + range(-20, -7) + range(-6, 0) + range(1, 7) + range(8, 21) + range(22, 27)) + self.fft_size / 2
        self.carrier = np.hstack((np.arange(-26, -21), np.arange(-20, -7), np.arange(-6, 0),
                                  np.arange(1, 7), np.arange(8, 21), np.arange(22, 27))) + self.fft_size // 2
        self.pilot_carrier = np.array([-21, -7, 7, 21]) + self.fft_size // 2
        self.pilot_symbol = np.array([1, 1, 1, -1])

        # self.tx_time_total = np.load('data/tx_time_1kP.npy')
        self.tx_time_total = np.load(folder_name + '/wifi_tx_time_cp.npy')
        # self.tx_time = self.tx_time_total[self.i_packet*self.N_total_frame:(self.i_packet+1)*self.N_total_frame, :]  # (pilot+data, fft)
        self.tx_time = self.tx_time_total
        # self.tx_time = self.tx_time * self.label_scale

        # self.tx_freq_total = np.load('data/tx_freq_1kP.npy')
        self.tx_freq_total = np.load(folder_name + '/wifi_tx_freq.npy')
        # self.tx_freq = self.tx_freq_total[self.i_packet*self.N_total_frame:(self.i_packet+1)*self.N_total_frame, :]
        self.tx_freq = self.tx_freq_total
        # self.tx_header_freq = np.load('data/tx_header_freq.npy')  # header contains 3 more symbols
        # self.tx_header_time = np.load('data/tx_header_time.npy')
        # self.tx_sync_time_cp, \
        # self.tx_data_time_cp, \
        # self.tx_time_cp = self.extract_pilots_data_add_cp_time(self.tx_time)
        self.tx_time_cp = self.tx_time

        self.tx_sync_freq = self.tx_freq[:self.N_sync_frame, :]
        self.tx_data_freq = self.tx_freq[self.N_sync_frame:, :]

        # self.rx_default_equ_freq = np.load('data/rx_default_equ.npy')
        # self.rx_header_default_equ_freq = np.load('data/rx_header_default_equ.npy')

        self.rx_time_cp = self.rx_time_cp_all[self.i_packet*self.N_total_frame:(self.i_packet+1)*self.N_total_frame, :]  # (pilot+data, fft+cp)
        # self.rx_time_cp = self.normalize(self.rx_time_cp)
        # self.rx_time_cp = self.rx_time_cp_all[536:536+94,:]

        self.rx_time_cp = self.rx_time_cp * self.input_scale
        #self.rx_time_cp = self.frequency_offset(self.rx_time_cp)
        self.rx_time_cp = self.add_noise(self.rx_time_cp, self.SNR)

        # self.rx_header_time = np.load('data/rx_header_raw_time.npy')  # (3, fft)
        # self.rx_header_time = self.rx_header_time * self.input_scale

        self.rx_sync_time_cp = self.rx_time_cp[:self.N_sync_frame, :]
        self.rx_data_time_cp = self.rx_time_cp[self.N_sync_frame:, :]

        # self.rx_time_default_equ = np.load('data/rx_default_equ.npy')
        # self.rx_freq_default_equ = np.load('data/rx_default_equ_freq.npy')
        # self.rx_freq_default_equ_pilot = self.rx_freq_default_equ[:self.N_pilot_frame, :]
        # self.rx_freq_default_equ_data = self.rx_freq_default_equ[
        #                                 self.N_pilot_frame:self.N_pilot_frame + self.N_data_frame, :]

        self.esn = ESN(
            n_inputs=self.window_size * 2,  # complex number to real
            n_outputs=2,
            n_reservoir=self.N_reservoir,
            spectral_radius=self.spectral_radius,
            sparsity=self.sparsity,
            input_scaling=self.reservoir_input_scale,
            noise=self.noise,
            teacher_forcing=False,
            teacher_scaling=self.label_scale,
            lut_activation=self.lut_activation,
            tanh_lut=self.tanh_lut,
            #out_activation=identity,
            #inverse_out_activation=identity,
        )

    def run(self):
        ### Directly demode raw rx data ###
        rx_sync_raw_freq = self.fft(self.rm_cp(self.rx_sync_time_cp))
        # ber_raw_sync = self.ber(rx_sync_raw_freq, self.tx_sync_freq)
        # if not self.silent:
        #     print('raw sync BER:', ber_raw_sync)
        #     self.plot_constellation(rx_sync_raw_freq, 'raw sync')  # pilot before rc
        #
        rx_data_raw_freq = self.fft(self.rm_cp(self.rx_data_time_cp))
        ber_raw_data = self.ber(rx_data_raw_freq[1:,:], self.tx_data_freq[1:,:])
        if not self.silent:
            print('raw data BER:', ber_raw_data)
        # # self.plot_constellation(rx_data_raw_freq, 'raw data')  # data before rc

        ### Default equalization in GNRradio###
        # ber_pilot_default = self.ber(self.rx_freq_default_equ_pilot, self.tx_pilot_freq)
        # ber_data_default = self.ber(self.rx_freq_default_equ_data, self.tx_data_freq)
        # print('default pilot BER:', ber_pilot_default)
        # print('default data BER:', ber_data_default)
        # self.plot_constellation(self.rx_freq_default_equ_pilot, 'default equ pilot')
        # self.plot_constellation(self.rx_freq_default_equ_data, 'default equ data')

        ### simple channel inverse ###
        channel = rx_sync_raw_freq[-2:, self.carrier] / self.tx_sync_freq[-2:, self.carrier]
        channel_time = self.ifft(channel)
        channel_mean = np.mean(channel[-2:,:], axis=0)
        if not self.silent:
            plt.plot(np.abs(channel_time.T), 'o-')
        # channel_mean = channel[-1,:]

        # simple_equ_pilot = copy.deepcopy(rx_pilot_raw_freq)
        # simple_equ_pilot[:, self.carrier] = simple_equ_pilot[:, self.carrier] / channel_mean
        # ber_data_simple = self.ber(simple_equ_pilot, self.tx_pilot_freq)
        # print('simple pilot BER:', ber_data_simple)
        # # self.plot_constellation(simple_equ_pilot, 'simple equ pilot')

        simple_equ_data = copy.deepcopy(rx_data_raw_freq[1:,:])
        simple_equ_data[:, self.carrier] = simple_equ_data[:, self.carrier] / channel_mean
        ber_data_LS = self.ber(simple_equ_data, self.tx_data_freq[1:,:])
        if not self.silent:
            print('LS data BER:', ber_data_LS)
            self.plot_constellation(simple_equ_data, 'LS equ data')

        ### dfe ###
        alpha = 0.1
        A = np.sqrt(2) / 2
        channel_mean_his = []
        sym_eq_set = []
        channel_mean_his.append(channel_mean)
        constellation = np.asarray([A + 1j * A, A - 1j * A, -A + 1j * A, -A - 1j * A])
        dfe_input_data = rx_data_raw_freq[1:, self.carrier]
        dfe_output_data = np.zeros((dfe_input_data.shape), dtype=complex)
        for n in range(dfe_input_data.shape[0]):
            sym_eq = dfe_input_data[n, :] / channel_mean
            sym_eq_set.append(sym_eq)
            sym_est = np.zeros(dfe_input_data.shape[1], dtype=complex)
            for m in range(sym_est.shape[0]):
                index = np.argmin(np.abs(constellation - sym_eq[m]))
                sym_est[m] = constellation[index]
            channel_mean = alpha * channel_mean + (1 - alpha) * (dfe_input_data[n, :] / sym_est)
            channel_mean_his.append(channel_mean)
            dfe_output_data[n, :] = sym_est
        simple_dfe_data = copy.deepcopy(rx_data_raw_freq[1:,:])
        simple_dfe_data[:, self.carrier] = dfe_output_data
        ber_data_simple_dfe = self.ber(simple_dfe_data, self.tx_data_freq[1:,:])
        if not self.silent:
            print('simple dfe data BER:', ber_data_simple_dfe)
            self.plot_constellation(np.asarray(sym_eq_set), 'simple dfe data', pilot_removed=True)

        ### Comb pilot interporlation ###
        alpha = 0.1
        channel_mean_comb = np.mean(channel[-2:, :], axis=0)
        Comb_input_data = rx_data_raw_freq[1:, :]
        Comb_output_data = np.zeros((Comb_input_data.shape), dtype=complex)
        for s in range(Comb_input_data.shape[0]):
            h_p = Comb_input_data[s, self.pilot_carrier] / self.tx_data_freq[s+1, self.pilot_carrier]
            h_p_avg = np.sum(h_p)
            h_p_6 = np.insert(h_p,[0,4],h_p_avg)
            f = interpolate.interp1d(np.asarray([0, 11, 25, 39, 53, 63]), h_p_6)
            xnew = np.arange(0, 64, 1)
            h = f(xnew)
            h = h[self.carrier]
            channel_mean_comb = alpha * channel_mean_comb + (1 - alpha) * h
            Comb_output_data[s,self.carrier] = Comb_input_data[s,self.carrier]/channel_mean_comb
        # Comb_data = copy.deepcopy(rx_data_raw_freq[1:, :])
        # Comb_data[:, self.carrier] = Comb_output_data
        ber_data_comb = self.ber(Comb_output_data, self.tx_data_freq[1:, :])
        if not self.silent:
            print('Comb data BER:', ber_data_comb)
            self.plot_constellation(Comb_output_data, 'Comb data', pilot_removed=False)

        ###STA###
        alpha = 0.1
        beta = 2
        channel_mean_sta = np.mean(channel[-2:, :], axis=0)
        sym_eq_set = []
        sta_input_data = rx_data_raw_freq[1:, self.carrier]
        sta_output_data = np.zeros((dfe_input_data.shape), dtype=complex)
        for n in range(sta_input_data.shape[0]):
            sym_eq = sta_input_data[n, :] / channel_mean_sta
            sym_eq_set.append(sym_eq)
            sym_est = np.zeros(sta_input_data.shape[1], dtype=complex)
            for m in range(sym_est.shape[0]):
                index = np.argmin(np.abs(constellation - sym_eq[m]))
                sym_est[m] = constellation[index]
            sta_output_data[n, :] = sym_est

            h_temp = sta_input_data[n, :] / sym_est
            h_64_temp = np.zeros(64,dtype=complex)
            h_p = rx_data_raw_freq[n+1, self.pilot_carrier] / self.tx_data_freq[n + 1, self.pilot_carrier]
            h_64_temp[self.carrier] = h_temp
            h_64_temp[self.pilot_carrier] = h_p
            h_update = np.zeros(64,dtype=complex)
            for i in range(6, 59):
                count = 0
                p_sum = 0 + 0j
                for j in range(i-beta,i+beta+1):
                    if j==32 or j<6 or j>58:
                        pass
                    else:
                        count += 1
                        p_sum = p_sum + h_64_temp[j]
                h_update[i] = p_sum/count
            h_update_carrier = h_update[self.carrier]
            channel_mean_sta = alpha * channel_mean_sta + (1 - alpha) * h_update_carrier
        sta_data = copy.deepcopy(rx_data_raw_freq[1:, :])
        sta_data[:, self.carrier] = sta_output_data
        ber_data_sta = self.ber(sta_data, self.tx_data_freq[1:, :])
        if not self.silent:
            print('sta data BER:', ber_data_sta)
            self.plot_constellation(np.asarray(sym_eq_set), 'sta data', pilot_removed=True)

        ### ESN with RLS and comb pilots ###
        tx_label_freq = np.zeros((self.N_total_frame, self.fft_size), dtype=complex)
        tx_label_freq[:, self.pilot_carrier] = self.tx_freq[:, self.pilot_carrier]
        tx_label_freq[:self.N_sync_frame, :] = self.tx_freq[:self.N_sync_frame, :]
        tx_label_time = self.ifft(tx_label_freq) # FIXME normalization?
        debug_freq = self.fft(tx_label_time)
        tx_label_time_CP = self.add_cp(tx_label_time)
        # self.train_and_test_RLS(self.rx_time_cp, tx_pilot_time_CP, train_with_CP=False)
        if self.method == 'RLS':
            BER = self.train_and_test_RLS(self.rx_time_cp, tx_label_time_CP)
        elif self.method == 'INV+RLS':
            BER = self.train_and_test_inv_RLS(self.rx_time_cp, tx_label_time_CP)
            #print('test\n' + 'rx_time_cp:' + self.rx_time_cp + '\ntx_label_time_CP:' + tx_label_time_CP)
            #print(tx_label_time_CP)

        ### ESN with RLS and decision feedback
        # self.train_and_test_RLS_DF(self.rx_time_cp, self.tx_time_cp)
        # self.train_and_test_inv_DF(self.rx_time_cp, self.tx_time_cp)

        ### ESN with Inv Sync symbols
        elif self.method == 'INV':
            BER = self.train_and_test_inv_sync(self.rx_time_cp, self.tx_time_cp)

        return BER, ber_data_simple_dfe, ber_data_LS, ber_data_comb, ber_data_sta
    # end of run()

    def train_and_test_inv_sync(self, rx_time_cp, tx_time_cp, train_with_CP=True):
        # recreate rx pilot time signal as input for training
        rx_time = self.rm_cp(rx_time_cp)

        # debug
        # self.plot_constellation(rx_pilot_freq[:, self.pilot_carrier], 'ESN input pilot', pilot_removed=True)
        # debug end

        if train_with_CP:
            esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
        else:
            esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time)
        # esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
        # esn_input_windowed = self.prepare_windowed_data_from_complex_block(np.concatenate((rx_time[:,-self.cp_len:],rx_time),axis=1))

        # generate label for training
        tx_time = self.rm_cp(tx_time_cp)
        if train_with_CP:
            esn_label_complex = tx_time_cp[:self.N_sync_frame, :].reshape((-1, 1))  # (N_symbols * (N_fft+N_cp), 1)
        else:
            esn_label_complex = tx_time[:self.N_sync_frame, :].reshape((-1, 1))
        esn_label = self.complex_to_real(esn_label_complex)  # (N_symbols * (N_fft+N_cp), 2)
        esn_label = np.concatenate((np.zeros((self.delay, 2)), esn_label), axis=0)

        # train ESN with INV on Sync symbols
        if train_with_CP:
            esn_input_inv = esn_input_windowed[:self.delay + self.N_sync_frame * (self.cp_len + self.fft_size), :]
            esn_label_inv = esn_label[:self.delay + self.N_sync_frame * (self.cp_len + self.fft_size), :]
        else:
            esn_input_inv = esn_input_windowed[:self.delay + self.N_sync_frame * self.fft_size, :]
            esn_label_inv = esn_label[:self.delay + self.N_sync_frame * self.fft_size, :]

        train_predicts = self.esn.train_inv(esn_input_inv, esn_label_inv, continuation=True)
        verify_predict_time = self.esn.test_inv(esn_input_inv, continuation=False)
        verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, self.N_sync_frame, cp_removed=False,
                                                         remove_delay=True)
        if not self.silent:
            self.plot_constellation(verify_predict_freq, 'ESN_inv_train_on_sync_symbols', pilot_removed=True)

        # test
        test_predict_time = self.esn.test_inv(esn_input_windowed, continuation=True)
        test_predict_freq = self.esn_output_to_block_f(test_predict_time, self.N_total_frame, cp_removed=False,
                                                         remove_delay=True)
        test_predict_freq_data = test_predict_freq[self.N_sync_frame+1:, :]
        ber_ESN = self.ber(test_predict_freq_data, self.tx_freq[self.N_sync_frame+1:, :])
        if not self.silent:
            self.plot_constellation(test_predict_freq_data, 'ESN_inv_test_data', pilot_removed=False)
            print('ESN BER:', ber_ESN)
        return ber_ESN

    def train_and_test_inv_DF(self,rx_time_cp, tx_time_cp):
        N_symbol = rx_time_cp.shape[0]
        len_symbol = self.fft_size + self.cp_len
        esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
        esn_label = tx_time_cp[:self.N_sync_frame, :]
        esn_label = self.complex_block_to_real_array(esn_label)
        esn_label = np.concatenate((np.zeros((self.delay, 2)), esn_label), axis=0)  # (N_samples + delay, 2)
        predict_freq_set = np.zeros((self.N_data_frame, self.fft_size), dtype=complex)

        for i in range(N_symbol):
            if i == self.N_sync_frame - 1:
                train_input = esn_input_windowed[0:(i + 1) * len_symbol + self.delay, :]
                train_label = esn_label[0:(i + 1) * len_symbol + self.delay, :]
                train_predicts = self.esn.train_inv(train_input, train_label, continuation=True)
                verify_predict_time = self.esn.test_inv(train_input, continuation=True)
                verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, 1, cp_removed=False,
                                                                 remove_delay=True)
                self.plot_constellation(verify_predict_freq, 'ESN_inv_train_verify', pilot_removed=True)
            elif i >= self.N_sync_frame:
                test_input = esn_input_windowed[i * len_symbol + self.delay:(i + 1) * len_symbol + self.delay, :]
                predicts = self.esn.test_inv(test_input, continuation=True)
                predicts_freq = self.esn_output_to_block_f(predicts, 1, cp_removed=False)
                self.plot_constellation(predicts_freq, 'ESN_inv_test', pilot_removed=True)
                recover_freq = self.recover_constellation(predicts_freq)
                # self.plot_constellation(recover_freq, 'ESN_inv_recover', pilot_removed=True)
                recover_time_array = self.complex_block_to_real_array(self.add_cp(self.ifft(recover_freq)))
                esn_label = np.concatenate((esn_label, recover_time_array), axis=0)  # save history for debug

                train_predicts = self.esn.train_inv(test_input, recover_time_array, continuation=True)

                predict_freq_set[i - self.N_sync_frame, :] = predicts_freq[0, :]
        ber_ESN_DF = self.ber(predict_freq_set, self.tx_freq[self.N_sync_frame:, :])
        print('ESN_inv_DF BER:', ber_ESN_DF)
        self.plot_constellation(predict_freq_set, 'ESN_inv_DF', pilot_removed=False)

    def train_and_test_RLS_DF(self,rx_time_cp, tx_time_cp):
        N_symbol = rx_time_cp.shape[0]
        len_symbol = self.fft_size + self.cp_len
        esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
        esn_label = tx_time_cp[:self.N_sync_frame, :]
        esn_label = self.complex_block_to_real_array(esn_label)
        esn_label = np.concatenate((np.zeros((self.delay, 2)), esn_label), axis=0)  #(N_samples + delay, 2)
        predict_freq_set = np.zeros((self.N_data_frame, self.fft_size), dtype=complex)

        for i in range(N_symbol):
            if i == 0:
                train_input = esn_input_windowed[i*len_symbol:(i+1)*len_symbol + self.delay, :]
                train_label = esn_label[i*len_symbol:(i+1)*len_symbol + self.delay, :]
                W_out, train_error, train_predicts, train_error_pre_w, train_predicts_pre_w = self.esn.train_RLS(
                    train_input, train_label, continuation=True)
                verify_predict_time = self.esn.test_RLS(train_input, W_out, continuation=True)
                verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, 1, cp_removed=False, remove_delay=True)
                # self.plot_constellation(verify_predict_freq, 'ESN_train_verify', pilot_removed=True)
            elif 0 < i < self.N_sync_frame:
                train_input = esn_input_windowed[i*len_symbol + self.delay:(i+1)*len_symbol + self.delay, :]
                train_label = esn_label[i*len_symbol + self.delay:(i+1)*len_symbol + self.delay, :]
                W_out, train_error, train_predicts, train_error_pre_w, train_predicts_pre_w = self.esn.train_RLS(
                    train_input, train_label, continuation=True)
                verify_predict_time = self.esn.test_RLS(train_input, W_out, continuation=True)
                verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, 1, cp_removed=False)
                # self.plot_constellation(verify_predict_freq, 'ESN_train_verify', pilot_removed=True)
            else:
                test_input = esn_input_windowed[i*len_symbol + self.delay:(i+1)*len_symbol + self.delay, :]
                predicts = self.esn.test_RLS(test_input, W_out[-1, :], continuation=True)
                predicts_freq = self.esn_output_to_block_f(predicts, 1, cp_removed=False)
                # self.plot_constellation(predicts_freq, 'ESN_test', pilot_removed=True)
                recover_freq = self.recover_constellation(predicts_freq)
                # self.plot_constellation(recover_freq, 'ESN_recover', pilot_removed=True)
                recover_time_array = self.complex_block_to_real_array(self.add_cp(self.ifft(recover_freq)))
                esn_label = np.concatenate((esn_label, recover_time_array), axis=0)  # save history for debug

                W_out, train_error, train_predicts, train_error_pre_w, train_predicts_pre_w = self.esn.train_RLS(
                    test_input, recover_time_array, continuation=True)
                predict_freq_set[i-self.N_sync_frame, :] = predicts_freq[0, :]
        ber_ESN_DF = self.ber(predict_freq_set, self.tx_freq[self.N_sync_frame:, :])
        print('ESN_DF BER:', ber_ESN_DF)

        self.plot_constellation(predict_freq_set, 'ESN_DF', pilot_removed=True)

    def train_and_test_RLS(self, rx_time_cp, tx_label_time_cp, train_with_CP=True):
        # recreate rx pilot time signal as input for training
        rx_time = self.rm_cp(rx_time_cp)
        rx_freq = self.fft(rx_time)
        rx_train_freq = np.zeros(rx_freq.shape, dtype=complex)
        rx_train_freq[:, self.pilot_carrier] = rx_freq[:, self.pilot_carrier]

        #debug
        # self.plot_constellation(rx_pilot_freq[:, self.pilot_carrier], 'ESN input pilot', pilot_removed=True)
        #debug end

        rx_train_time = self.ifft(rx_train_freq)
        rx_train_time[:self.N_sync_frame, :] = rx_time[:self.N_sync_frame, :]
        rx_train_time_cp = self.add_cp(rx_train_time)  # (N_symbols, N_fft+N_cp)
        rx_train_time_cp[:self.N_sync_frame, :] = rx_time_cp[:self.N_sync_frame, :]
        if train_with_CP:
            esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_train_time_cp)
        else:
            esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_train_time)
        # esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
        # esn_input_windowed = self.prepare_windowed_data_from_complex_block(np.concatenate((rx_time[:,-self.cp_len:],rx_time),axis=1))

        # generate tx pilot time signal as label for training
        tx_label_time = self.rm_cp(tx_label_time_cp)
        if train_with_CP:
            esn_label_complex = tx_label_time_cp.reshape((-1, 1))  # (N_symbols * (N_fft+N_cp), 1)
        else:
            esn_label_complex = tx_label_time.reshape((-1, 1))
        esn_label = self.complex_to_real(esn_label_complex)  # (N_symbols * (N_fft+N_cp), 2)
        esn_label = np.concatenate((np.zeros((self.delay, 2)), esn_label), axis=0)
        debug_tx_freq = self.fft(tx_label_time)

        # train ESN
        np.save('gt_train_input_{}.npy'.format(self.i_packet), esn_input_windowed)
        np.save('gt_train_label_{}.npy'.format(self.i_packet), esn_label)
        W_out, train_error, train_predicts, train_error_pre_w, train_predicts_pre_w = self.esn.train_RLS(esn_input_windowed, esn_label, continuation=False)
        np.save('gt_train_pred.npy', train_predicts)
        train_predicts_freq = self.esn_output_to_block_f(train_predicts, self.N_total_frame, cp_removed=not train_with_CP, remove_delay=True)
        if not self.silent:
            self.plot_constellation(train_predicts_freq, 'ESN_train', pilot_removed=True)

        #debug
        # train_predicts_freq_2 = self.esn_output_to_block_f(train_predicts_pre_w, N_symbol, cp_removed=not train_with_CP,remove_delay=True)
        # self.plot_constellation(train_predicts_freq_2, 'ESN_train_pre', pilot_removed=True)
        #end debug

        # verify W_out with training data
        verify_predict_time = self.esn.test_RLS(esn_input_windowed, W_out, self.i_packet, continuation=False)
        import torch
        loss_fn = torch.nn.MSELoss()
        
        verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, self.N_total_frame, cp_removed=not train_with_CP, remove_delay=True)
        if not self.silent:
            self.plot_constellation(verify_predict_freq, 'ESN_train_verify', pilot_removed=True)

        # test ESN
        if train_with_CP:
            test_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
        else:
            test_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time)
        predicts_time = self.esn.test_RLS(test_input_windowed, W_out, self.i_packet, continuation=False,
                                          debug=self.debug, output_folder=self.output_folder)  # (N_symbols * (N_fft+N_cp), 2)
        np.save('gt_test_input_{}.npy'.format(self.i_packet), test_input_windowed)
        # predicts_time_complex = self.real_to_complex(predicts_time)  # (N_symbols * (N_fft+N_cp), 1)
        # prdicts_time_reshape = predicts_time_complex.reshape(rx_time_cp.shape)
        # predicts_time_noCP = self.rm_cp(prdicts_time_reshape)
        # predicts_freq = self.fft(predicts_time_noCP)
    
        np.save('gt_time.npy', predicts_time)
        predicts_freq = self.esn_output_to_block_f(predicts_time, self.N_total_frame, cp_removed=not train_with_CP, remove_delay=True)
        np.save('gt_freq.npy', predicts_freq)
        ber_ESN = self.ber(predicts_freq[self.N_sync_frame+1:, :], self.tx_freq[self.N_sync_frame+1:, :])
        X = self.tx_freq[self.N_sync_frame+1:, :]
        if not self.silent:
            print('ESN BER:', ber_ESN)
            self.plot_constellation(predicts_freq[:self.N_sync_frame, :], 'ESN_pilot', pilot_removed=True)
            self.plot_constellation(predicts_freq[self.N_sync_frame:, :], 'ESN_data', pilot_removed=False)

        import torch 
        loss_fn = torch.nn.MSELoss()
        
        def R2P(a):
            a_real = np.real(a)
            a_imag = np.imag(a)
            return np.concatenate((a_real, a_imag), axis=0)
       
        pred = R2P(predicts_freq[self.N_sync_frame+1:, :]) 
        label = R2P(self.tx_freq[self.N_sync_frame+1:, :])


        mse_loss = (np.square(pred - label)).mean(axis=None)

        return ber_ESN

    def train_and_test_inv_RLS(self, rx_time_cp, tx_label_time_cp, train_with_CP=True):
        #print("input1:", rx_time_cp.shape)
        #print("input2:", tx_label_time_cp.shape)
        # recreate rx pilot time signal as input for training
        rx_time = self.rm_cp(rx_time_cp)
        #print('rx_time:', rx_time.shape)
        rx_freq = self.fft(rx_time)
        #print('rx_freq:', rx_freq.shape)
        rx_train_freq = np.zeros(rx_freq.shape, dtype=complex)
        #print('rx_train_freq:', rx_train_freq.shape)
        rx_train_freq[:, self.pilot_carrier] = rx_freq[:, self.pilot_carrier]
        #print('rx_train_freq[:,self.pilot_carrier]:', rx_train_freq[:, self.pilot_carrier].shape)

        #debug
        if not self.silent:
            self.plot_constellation(rx_train_freq, 'ESN input pilots', pilot_removed=True)
        #debug end

        rx_train_time = self.ifft(rx_train_freq)
        rx_train_time[:self.N_sync_frame, :] = rx_time[:self.N_sync_frame, :]
        rx_train_time_cp = self.add_cp(rx_train_time)  # (N_symbols, N_fft+N_cp)
        rx_train_time_cp[:self.N_sync_frame, :] = rx_time_cp[:self.N_sync_frame, :]
        if train_with_CP:
            esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_train_time_cp)
        else:
            esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_train_time)
        # esn_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
        # esn_input_windowed = self.prepare_windowed_data_from_complex_block(np.concatenate((rx_time[:,-self.cp_len:],rx_time),axis=1))

        # generate label for training
        tx_label_time = self.rm_cp(tx_label_time_cp)
        if train_with_CP:
            esn_label_complex = tx_label_time_cp.reshape((-1, 1))  # (N_symbols * (N_fft+N_cp), 1)
        else:
            esn_label_complex = tx_label_time.reshape((-1, 1))
        esn_label = self.complex_to_real(esn_label_complex)  # (N_symbols * (N_fft+N_cp), 2)
        esn_label = np.concatenate((np.zeros((self.delay, 2)), esn_label), axis=0)
        debug_tx_freq = self.fft(tx_label_time)

        # train ESN with INV on Sync symbols
        if train_with_CP:
            esn_input_inv = esn_input_windowed[:self.delay + self.N_sync_frame * (self.cp_len + self.fft_size), :]
            esn_label_inv = esn_label[:self.delay + self.N_sync_frame * (self.cp_len + self.fft_size), :]
        else:
            esn_input_inv = esn_input_windowed[:self.delay + self.N_sync_frame * self.fft_size, :]
            esn_label_inv = esn_label[:self.delay + self.N_sync_frame * self.fft_size, :]

        train_predicts = self.esn.train_inv(esn_input_inv, esn_label_inv, continuation=True)
        verify_predict_time = self.esn.test_inv(esn_input_inv, continuation=True)
        verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, self.N_sync_frame, cp_removed=False,
                                                         remove_delay=True)
        #if not self.silent:
            #self.plot_constellation(verify_predict_freq, 'ESN_inv_train_on_sync_symbols', pilot_removed=True)
        # print rrmse
        #rrmse = self.rrmse(train_predicts, esn_label_inv)
        #print('INV relative rmse:', rrmse)

        # train ESN_with_RLS on data symbols
        if train_with_CP:
            esn_input_RLS = esn_input_windowed[self.delay + self.N_sync_frame * (self.cp_len + self.fft_size):, :]
            esn_label_RLS = esn_label[self.delay + self.N_sync_frame * (self.cp_len + self.fft_size):, :]
        else:
            esn_input_RLS = esn_input_windowed[self.delay + self.N_sync_frame * self.fft_size:, :]
            esn_label_RLS = esn_label[self.delay + self.N_sync_frame * self.fft_size:, :]

        W_out, train_error, train_predicts, train_error_pre_w, train_predicts_pre_w = self.esn.train_RLS(esn_input_RLS, esn_label_RLS, continuation=True)
        train_predicts_freq = self.esn_output_to_block_f(train_predicts, self.N_data_frame, cp_removed=not train_with_CP, remove_delay=False)
        #if not self.silent:
            #self.plot_constellation(train_predicts_freq, 'ESN_RLS_train_data_symbols', pilot_removed=True)

        # print rrmse
        #rrmse = self.rrmse(train_predicts, esn_label_RLS)
        #print('RLS relative rmse:', rrmse)

        #debug
        # train_predicts_freq_2 = self.esn_output_to_block_f(train_predicts_pre_w, N_symbol, cp_removed=not train_with_CP,remove_delay=True)
        # self.plot_constellation(train_predicts_freq_2, 'ESN_train_pre', pilot_removed=True)
        #end debug

        # verify W_out with training data
        if self.use_fpga:
            verify_predict_time = self.esn.test_RLS_FPGA(self.sock, self.addr, esn_input_RLS, W_out, continuation=False)
        else:
            verify_predict_time = self.esn.test_RLS(esn_input_RLS, W_out, self.i_packet, continuation=False)
        verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, self.N_data_frame, cp_removed=not train_with_CP, remove_delay=False)
        #if not self.silent:
            #self.plot_constellation(verify_predict_freq, 'ESN_RLS_train_verify', pilot_removed=True)

        # test ESN with RLS on data symbols
        if train_with_CP:
            test_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
            test_input_RLS = test_input_windowed[self.delay + self.N_sync_frame * (self.cp_len + self.fft_size):, :]
        else:
            test_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time)
            test_input_RLS = test_input_windowed[self.delay + self.N_sync_frame * self.fft_size:, :]
        if self.use_fpga:
            predicts_time = self.esn.test_RLS_FPGA(self.sock, self.addr, test_input_RLS, W_out, continuation=True)  # (N_symbols * (N_fft+N_cp), 2)
        else:
            predicts_time = self.esn.test_RLS(test_input_RLS, W_out, self.i_packet, continuation=True, debug=self.debug, output_folder=self.output_folder)  # (N_symbols * (N_fft+N_cp), 2)

        # predicts_time_complex = self.real_to_complex(predicts_time)  # (N_symbols * (N_fft+N_cp), 1)
        # prdicts_time_reshape = predicts_time_complex.reshape(rx_time_cp.shape)
        # predicts_time_noCP = self.rm_cp(prdicts_time_reshape)
        # predicts_freq = self.fft(predicts_time_noCP)
        # print('predicts_time:', predicts_time)
        predicts_freq = self.esn_output_to_block_f(predicts_time, self.N_data_frame, cp_removed=not train_with_CP, remove_delay=False)
        # print('predicts_freq:', predicts_freq[1:,:].shape, predicts_freq[1:,:])
        # print('tx_freq:', self.tx_freq.shape,self.tx_freq)
        # print('tx_freq_part:', self.tx_freq[self.N_sync_frame+1:, :].shape)
        ber_ESN = self.ber(predicts_freq[1:,:], self.tx_freq[self.N_sync_frame+1:, :])
        if True: ##not self.silent:
            print('ESN BER:', ber_ESN)
            #self.plot_constellation(predicts_freq, 'ESN_data', pilot_removed=False)

        if self.show_wout:
            plt.figure()
            plt.plot(W_out[0, 0, :])
            plt.plot(W_out[0, 1, :])
            plt.title('ESN W_out RLS')
            plt.show()

        '''
        #### TODO:ADD TEST_RLS_HW
        predicts_time_hw = self.esn.predict_hw(test_input_RLS, W_out, continuation=False, debug=self.debug)
        predicts_time_hw = np.clip(predicts_time_hw, a_min=-8, a_max=8)
        predicts_freq_hw = self.esn_output_to_block_f(predicts_time_hw, self.N_data_frame, cp_removed=not train_with_CP, remove_delay=False)
        ber_ESN_hw = self.ber(predicts_freq_hw[1:,:], self.tx_freq[self.N_sync_frame+1:, :])
        print('ESN HW BER:', ber_ESN_hw)
        self.plot_constellation(predicts_freq_hw, 'ESN_HW', pilot_removed=False)

        # Save HW prediction
        fname_head = self.output_folder + str(self.i_packet)
        if self.savedat:
            np.savetxt(fname_head + '/predict_py.dat', predicts_time.flatten(), delimiter='\n', fmt='%10f')
            np.savetxt(fname_head + '/predict_hw_sim.dat', predicts_time_hw.flatten(), delimiter='\n', fmt='%10f')

        # DIFF
        diff_hw = np.abs(predicts_time_hw - predicts_time)
        print('HW sim diff avg: ', np.mean(diff_hw), diff_hw)
        if self.debug:
            plt.figure()
            plt.plot(diff_hw[:, 0])
            plt.plot(diff_hw[:, 1])
            plt.title('HW sim diff')
            plt.show()
        '''

        return ber_ESN #, ber_ESN_hw

    def recover_constellation(self, d):
        A = np.sqrt(2) / 2
        constellation = np.asarray([A + 1j * A, A - 1j * A, -A + 1j * A, -A - 1j * A])
        d_recover = np.zeros(d.shape, dtype=complex)
        d_recover[:, self.pilot_carrier] = self.pilot_symbol
        d_data = d[:,self.carrier]
        for i in range(d_data.shape[0]):
            for j in range(d_data.shape[1]):
                index = np.argmin(np.abs(constellation - d_data[i,j]))
                d_data[i,j] = constellation[index]
        d_recover[:, self.carrier] = d_data
        return d_recover

    def complex_block_to_real_array(self, d):
        d_reshape = d.reshape((-1, 1))
        d_real = self.complex_to_real(d_reshape)
        return d_real

    def esn_output_to_block_f(self, d, N_symbol, cp_removed=False, remove_delay=False):
        if remove_delay:
            d = d[self.delay:, :]
        d_complex = self.real_to_complex(d)  # (N_symbols * (N_fft+N_cp), 1)
        d_reshape = d_complex.reshape((N_symbol, -1))
        if cp_removed:
            d_noCP = d_reshape
        else:
            d_noCP = self.rm_cp(d_reshape)
        d_freq = self.fft(d_noCP)
        return d_freq

    def prepare_windowed_data_from_complex_block(self, block):
        """
        input complex block : (N_symbols, symbol_len) e.g. symbol_len = N_fft+N_cp or N_fft
        output real array_windowed : (N_symbols * symbol_len, 2 * N_window)
        """
        array_complex = block.reshape((-1, 1))  # (N_symbols * (N_fft+N_cp), 1)
        array = np.concatenate((np.real(array_complex), np.imag(array_complex)), axis=1) # (N_symbols * (N_fft+N_cp), 2)
        # generate windowed input
        array_extended = np.concatenate((np.zeros((self.window_size-1, 2)), array, np.zeros((self.delay, 2))), axis=0)  # (Delay + N_window - 1 + N_symbols * (N_fft+N_cp), 2)
        array_windowed = np.zeros((array.shape[0] + self.delay, self.window_size * 2)) # (N_symbols * (N_fft+N_cp) + delay, 2 * N_window)
        for i in range(array.shape[0] + self.delay):
            array_windowed[i, :] = array_extended[i:i+self.window_size, :].reshape(-1)
        return array_windowed  # (N_symbols * (N_fft+N_cp) + delay, 2 * N_window)

    def add_cp(self, d):
        d_cp = np.concatenate((d[:, -self.cp_len:], d), axis=1)
        return d_cp

    def rm_cp(self, d_cp):
        d = d_cp[:, self.cp_len:]
        return d

    def complex_to_real(self, c):
        r = np.concatenate((np.real(c), np.imag(c)), axis=1)
        return r

    def real_to_complex(self, r):
        c = r[:, 0] + 1j * r[:, 1]
        return c

    def extract_pilots_data_add_cp_time(self, tx_time):
        pilots = tx_time[:self.N_sync_frame, :]
        data = tx_time[self.N_sync_frame:self.N_sync_frame + self.N_data_frame, :]
        pilots_cp = np.concatenate((pilots[:, -self.cp_len:], pilots), axis=1)
        data_cp = np.concatenate((data[:, -self.cp_len:], data), axis=1)
        total_cp = np.concatenate((tx_time[:, -self.cp_len:], tx_time), axis=1)
        return pilots_cp, data_cp, total_cp

    def fft(self, t):
        ##f = np.fft.fft(t, axis=1) / float(self.fft_size)
        f = np.fft.fft(t, axis=1) / np.sqrt(self.fft_size)
        f_shifted = np.concatenate((f[:, self.fft_size // 2:], f[:, :self.fft_size // 2]),
                                   axis=1)  # this align with gnuradio 'shifted' option
        return f_shifted

    def ifft(self, f_shifted):
        f = np.concatenate((f_shifted[:, self.fft_size // 2:], f_shifted[:, :self.fft_size // 2]),
                           axis=1)  # this align with gnuradio 'shifted' option
        ##t = np.fft.ifft(f, axis=1) * self.fft_size
        t = np.fft.ifft(f, axis=1) * np.sqrt(self.fft_size)
        return t

    def ber(self, a, b):
        assert a.shape == b.shape
        a_list = a[:, self.carrier].reshape((1, -1))
        b_list = b[:, self.carrier].reshape((1, -1))

        #a_list_2 = a[:, self.carrier].reshape((-1, 1))
        #b_list_2 = b[:, self.carrier].reshape((-1, 1))
        #print('len_a:', len(a_list_2))
        #print('len_b:', len(b_list_2))


        # with open('data/check/a_list.txt', 'w') as f1, open('data/check/b_list.txt', 'w') as f2:
        #     for i in range(len(a_list_2)):
        #         f1.write(str(a_list_2[i]) + '\n')
        #     for j in range(len(b_list_2)):
        #         f2.write(str(b_list_2[j]) + '\n')

        a_bits = self.qpsk_demod(a_list)
        b_bits = self.qpsk_demod(b_list)

        # a_bits_str = a_bits.tostring()
        # b_bits_str = b_bits.tostring()
        # print('a_str:',a_bits_str[0:100])
        # print('b_str', b_bits_str[0:100])
        # if a_bits_str == b_bits_str:
        #     print('same')


        # print('a_bits:', len(a_bits))
        # print('b_bits:', len(b_bits))
        # with open('data/check/a_bits.txt', 'w+') as f1, open('data/check/b_bits.txt', 'w+') as f2:
        #     for i in range(len(a_bits)):
        #         f1.write(str(a_bits[i]) + '\n')
        #     for j in range(len(b_bits)):
        #         f2.write(str(b_bits[j]) + '\n')

        error_count = np.sum(a_bits != b_bits)
        total_bits = a_bits.shape[1]
        #print('error_count: ', str(error_count))
        #print('total_bits: ', str(total_bits))
        ber = float(error_count) / float(total_bits)
        return ber

    def qpsk_demod(self, a):
        a_real = np.real(a)
        a_imag = np.imag(a)

        a_real_bit = np.ones(a_real.shape)
        a_real_bit[a_real < 0] = 0

        a_imag_bit = np.ones(a_imag.shape)
        a_imag_bit[a_imag < 0] = 0

        a_bit = np.concatenate((a_imag_bit, a_real_bit), axis=0)
        a_bit = a_bit.reshape((1, -1), order='F')
        return a_bit

    def plot_constellation(self, a, title='', pilot_removed=False):
        if pilot_removed:
            pass
        else:
            a = a[:, self.carrier].reshape(-1)
        a_real = np.real(a)
        a_imag = np.imag(a)
        plt.figure()
        plt.scatter(a_real, a_imag)
        plt.title(title)
        plt.show()

    def add_noise(self, d, snr):
        noise_var_real = np.mean(np.real(d) ** 2) * 10 ** (-float(snr) / 10)
        noise_var_imag = np.mean(np.imag(d) ** 2) * 10 ** (-float(snr) / 10)
        noise = np.random.normal(0, np.sqrt(noise_var_real), d.shape) + 1j * np.random.normal(0, np.sqrt(
            noise_var_imag), d.shape)
        noisy_d = d + noise
        return noisy_d

    def remove_zeros(self, d):
        p = np.var(d[0,:]) # np.sum(np.abs(d[0,:]))
        # index = np.sum(np.abs(d),axis=1) > p/10.0
        index = np.var(d, axis=1) > 0.5 * p
        d_no_zero = d[index, :]
        return d_no_zero

    def normalize(self,d):
        mean = np.mean(d)
        std = np.std(d)
        d_normal = (d-mean)/std
        return d_normal

    def frequency_offset(self,d):
        sts_160 = d[:2].reshape(-1)
        lts_160 = d[2:4].reshape(-1)
        lts_128 = lts_160[-128:]

        sum_sts = 0
        for i in range(160-16):
            sum_sts += sts_160[i] * np.conj(sts_160[i+16])
        sts_ang = 1.0/16.0 * np.angle(sum_sts)

        # FIXME issues with lts_ang computation
        sum_lts = 0
        for j in range(64):
            sum_lts += lts_128[j] * np.conj(lts_128[j+64])
        lts_ang = 1.0/64.0 * np.angle(sum_lts)

        sum_lts_32 = 0
        for k in range(32):
            sum_lts_32 += lts_160[k] * np.conj(lts_160[k+128])
        lts_32_ang = 1.0/128.0 * np.angle(sum_lts_32)

        print("STS frequency offset ", sts_ang, 'LTS frequency offset ', lts_ang, lts_32_ang)

        # FIXME better accuracy required
        #freq_comp = (lts_32_ang+lts_ang+sts_ang)/3.0
        #freq_comp = (lts_32_ang + lts_ang) / 2.0
        freq_comp = lts_32_ang
        d_copy = d.reshape(-1)
        for i in range(len(d_copy)):
            d_copy[i] = d_copy[i] * np.exp(1j * float(i) * freq_comp)
        return d_copy.reshape(d.shape)

    def rrmse(self, inference, label):
        inference_flat = inference.flatten()
        label_flat = label.flatten()
        small_lamb = 1e-6
        rdiff = np.divide((inference_flat - label_flat), (label_flat + small_lamb))  # relative difference
        return np.sqrt(rdiff.dot(rdiff) / rdiff.size)
