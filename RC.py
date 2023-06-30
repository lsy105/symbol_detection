import numpy as np
import matplotlib.pyplot as plt
import copy
from pyESN import ESN

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
            N_reservoir=32,
    ):
        self.silent = silent

        self.input_scale = 100
        self.label_scale = 1

        self.method = method
        self.i_packet = 0
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
        self.spectral_radius = 0.2
        self.sparsity = 0.4
        self.noise = 1e-6
        # folder_name = 'data/WIFI/5890M1MV2'
        # folder_name = 'data/WIFI/5890M200k'

        # self.rx_time_cp_all = np.load('data/rx_cp_raw_time_200k_Ant.npy')
        self.rx_time_cp_all = np.load('data/rx_cp_raw_time_200k_Ant_NS.npy')
        # self.rx_time_cp_all = np.load('data/rx_cp_raw_time_200k_Ant2_1kP.npy')
        # self.rx_time_cp_all = np.load(folder_name + '/wifi_rx_time_cp.npy')

        self.carrier = np.array(range(-26, -21) + range(-20, -7) + range(-6, 0) + range(1, 7) + range(8, 21) + range(22,
                                                                                                                     27)) + self.fft_size / 2
        self.pilot_carrier = np.array([-21, -7, 7, 21]) + self.fft_size / 2
        # self.pilot_carrier = np.array(range(-32, 32, 2)) + self.fft_size / 2
        # self.pilot_carrier = np.array(range(-26, 27, 2)) + self.fft_size / 2
        self.pilot_symbol = np.array([1, 1, 1, -1])

        self.tx_time_total = np.load('data/tx_time_1kP.npy')
        # self.tx_time_total = np.load(folder_name + '/wifi_tx_time_cp.npy')
        self.tx_time = self.tx_time_total[self.i_packet*self.N_total_frame:(self.i_packet+1)*self.N_total_frame, :]  # (pilot+data, fft)
        self.tx_time = self.tx_time * self.label_scale

        self.tx_freq_total = np.load('data/tx_freq_1kP.npy')
        # self.tx_freq_total = np.load(folder_name + '/wifi_tx_freq.npy')
        self.tx_freq = self.tx_freq_total[self.i_packet*self.N_total_frame:(self.i_packet+1)*self.N_total_frame, :]
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
        # self.rx_time_cp = self.rx_time_cp_all[536:536+94,:]

        self.rx_time_cp = self.rx_time_cp * self.input_scale
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
            noise=self.noise,
            teacher_forcing=False
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
        ber_raw_data = self.ber(rx_data_raw_freq, self.tx_data_freq)
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
        plt.plot(np.abs(channel_time.T), 'o-')
        # channel_mean = channel[-1,:]

        # simple_equ_pilot = copy.deepcopy(rx_pilot_raw_freq)
        # simple_equ_pilot[:, self.carrier] = simple_equ_pilot[:, self.carrier] / channel_mean
        # ber_data_simple = self.ber(simple_equ_pilot, self.tx_pilot_freq)
        # print('simple pilot BER:', ber_data_simple)
        # # self.plot_constellation(simple_equ_pilot, 'simple equ pilot')

        simple_equ_data = copy.deepcopy(rx_data_raw_freq)
        simple_equ_data[:, self.carrier] = simple_equ_data[:, self.carrier] / channel_mean
        ber_data_simple = self.ber(simple_equ_data, self.tx_data_freq)
        if not self.silent:
            print('LS data BER:', ber_data_simple)
            self.plot_constellation(simple_equ_data, 'LS equ data')

        ### dfe ###
        alpha = 0.1
        A = np.sqrt(2) / 2
        channel_mean_his = []
        sym_eq_set = []
        channel_mean_his.append(channel_mean)
        constellation = np.asarray([A + 1j * A, A - 1j * A, -A + 1j * A, -A - 1j * A])
        dfe_input_data = rx_data_raw_freq[:, self.carrier]
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
        simple_dfe_data = copy.deepcopy(rx_data_raw_freq)
        simple_dfe_data[:, self.carrier] = dfe_output_data
        ber_data_simple_dfe = self.ber(simple_dfe_data, self.tx_data_freq)
        if not self.silent:
            print('simple dfe data BER:', ber_data_simple_dfe)
            self.plot_constellation(np.asarray(sym_eq_set), 'simple dfe data', pilot_removed=True)


        ### ESN with RLS and comb pilots ###
        tx_label_freq = np.zeros((self.N_total_frame, self.fft_size), dtype=complex)
        tx_label_freq[:, self.pilot_carrier] = self.tx_freq[:, self.pilot_carrier]
        tx_label_freq[:self.N_sync_frame, :] = self.tx_freq[:self.N_sync_frame, :]
        tx_label_time = self.ifft(tx_label_freq)
        debug_freq = self.fft(tx_label_time)
        tx_label_time_CP= self.add_cp(tx_label_time)
        # self.train_and_test_RLS(self.rx_time_cp, tx_pilot_time_CP, train_with_CP=False)
        if self.method == 'RLS':
            BER = self.train_and_test_RLS(self.rx_time_cp, tx_label_time_CP)
        elif self.method == 'INV+RLS':
            BER = self.train_and_test_inv_RLS(self.rx_time_cp, tx_label_time_CP)

        ### ESN with RLS and decision feedback
        # self.train_and_test_RLS_DF(self.rx_time_cp, self.tx_time_cp)
        # self.train_and_test_inv_DF(self.rx_time_cp, self.tx_time_cp)

        ### ESN with Inv Sync symbols
        elif self.method == 'INV':
            BER = self.train_and_test_inv_sync(self.rx_time_cp, self.tx_time_cp)

        return BER, ber_data_simple_dfe, ber_data_simple
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
        test_predict_freq_data = test_predict_freq[self.N_sync_frame:, :]
        ber_ESN = self.ber(test_predict_freq_data, self.tx_freq[self.N_sync_frame:, :])
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
        W_out, train_error, train_predicts, train_error_pre_w, train_predicts_pre_w = self.esn.train_RLS(esn_input_windowed, esn_label, continuation=False)
        train_predicts_freq = self.esn_output_to_block_f(train_predicts, self.N_total_frame, cp_removed=not train_with_CP, remove_delay=True)
        if not self.silent:
            self.plot_constellation(train_predicts_freq, 'ESN_train', pilot_removed=True)

        #debug
        # train_predicts_freq_2 = self.esn_output_to_block_f(train_predicts_pre_w, N_symbol, cp_removed=not train_with_CP,remove_delay=True)
        # self.plot_constellation(train_predicts_freq_2, 'ESN_train_pre', pilot_removed=True)
        #end debug

        # verify W_out with training data
        verify_predict_time = self.esn.test_RLS(esn_input_windowed, W_out, continuation=False)
        verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, self.N_total_frame, cp_removed=not train_with_CP, remove_delay=True)
        if not self.silent:
            self.plot_constellation(verify_predict_freq, 'ESN_train_verify', pilot_removed=True)


        # test ESN
        if train_with_CP:
            test_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
        else:
            test_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time)
        predicts_time = self.esn.test_RLS(test_input_windowed, W_out, continuation=False)  # (N_symbols * (N_fft+N_cp), 2)
        # predicts_time_complex = self.real_to_complex(predicts_time)  # (N_symbols * (N_fft+N_cp), 1)
        # prdicts_time_reshape = predicts_time_complex.reshape(rx_time_cp.shape)
        # predicts_time_noCP = self.rm_cp(prdicts_time_reshape)
        # predicts_freq = self.fft(predicts_time_noCP)
        predicts_freq = self.esn_output_to_block_f(predicts_time, self.N_total_frame, cp_removed=not train_with_CP, remove_delay=True)
        ber_ESN = self.ber(predicts_freq[self.N_sync_frame:, :], self.tx_freq[self.N_sync_frame:, :])
        if not self.silent:
            print('ESN BER:', ber_ESN)
            self.plot_constellation(predicts_freq[:self.N_sync_frame, :], 'ESN_pilot', pilot_removed=True)
            self.plot_constellation(predicts_freq[self.N_sync_frame:, :], 'ESN_data', pilot_removed=False)
        return ber_ESN

    def train_and_test_inv_RLS(self, rx_time_cp, tx_label_time_cp, train_with_CP=True):
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
        if not self.silent:
            self.plot_constellation(verify_predict_freq, 'ESN_inv_train_on_sync_symbols', pilot_removed=True)


        # train ESN_with_RLS on data symbols
        if train_with_CP:
            esn_input_RLS = esn_input_windowed[self.delay + self.N_sync_frame * (self.cp_len + self.fft_size):, :]
            esn_label_RLS = esn_label[self.delay + self.N_sync_frame * (self.cp_len + self.fft_size):, :]
        else:
            esn_input_RLS = esn_input_windowed[self.delay + self.N_sync_frame * self.fft_size:, :]
            esn_label_RLS = esn_label[self.delay + self.N_sync_frame * self.fft_size:, :]

        W_out, train_error, train_predicts, train_error_pre_w, train_predicts_pre_w = self.esn.train_RLS(esn_input_RLS, esn_label_RLS, continuation=True)
        train_predicts_freq = self.esn_output_to_block_f(train_predicts, self.N_data_frame, cp_removed=not train_with_CP, remove_delay=False)
        if not self.silent:
            self.plot_constellation(train_predicts_freq, 'ESN_RLS_train_data_symbols', pilot_removed=True)

        #debug
        # train_predicts_freq_2 = self.esn_output_to_block_f(train_predicts_pre_w, N_symbol, cp_removed=not train_with_CP,remove_delay=True)
        # self.plot_constellation(train_predicts_freq_2, 'ESN_train_pre', pilot_removed=True)
        #end debug

        # verify W_out with training data
        verify_predict_time = self.esn.test_RLS(esn_input_RLS, W_out, continuation=False)
        verify_predict_freq = self.esn_output_to_block_f(verify_predict_time, self.N_data_frame, cp_removed=not train_with_CP, remove_delay=False)
        if not self.silent:
            self.plot_constellation(verify_predict_freq, 'ESN_RLS_train_verify', pilot_removed=True)


        # test ESN with RLS on data symbols
        if train_with_CP:
            test_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time_cp)
            test_input_RLS = test_input_windowed[self.delay + self.N_sync_frame * (self.cp_len + self.fft_size):, :]
        else:
            test_input_windowed = self.prepare_windowed_data_from_complex_block(rx_time)
            test_input_RLS = test_input_windowed[self.delay + self.N_sync_frame * self.fft_size:, :]
        predicts_time = self.esn.test_RLS(test_input_RLS, W_out, continuation=True)  # (N_symbols * (N_fft+N_cp), 2)
        # predicts_time_complex = self.real_to_complex(predicts_time)  # (N_symbols * (N_fft+N_cp), 1)
        # prdicts_time_reshape = predicts_time_complex.reshape(rx_time_cp.shape)
        # predicts_time_noCP = self.rm_cp(prdicts_time_reshape)
        # predicts_freq = self.fft(predicts_time_noCP)
        predicts_freq = self.esn_output_to_block_f(predicts_time, self.N_data_frame, cp_removed=not train_with_CP, remove_delay=False)
        ber_ESN = self.ber(predicts_freq, self.tx_freq[self.N_sync_frame:, :])
        if not self.silent:
            print('ESN BER:', ber_ESN)
            self.plot_constellation(predicts_freq, 'ESN_data', pilot_removed=False)

        return ber_ESN


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
        f = np.fft.fft(t, axis=1) / float(self.fft_size)
        f_shifted = np.concatenate((f[:, self.fft_size / 2:], f[:, :self.fft_size / 2]),
                                   axis=1)  # this align with gnuradio 'shifted' option
        return f_shifted

    def ifft(self, f_shifted):
        f = np.concatenate((f_shifted[:, self.fft_size / 2:], f_shifted[:, :self.fft_size / 2]),
                           axis=1)  # this align with gnuradio 'shifted' option
        t = np.fft.ifft(f, axis=1) * self.fft_size
        return t

    def add_cp(self, a):
        a_cp = np.concatenate((a[:, -self.cp_len:], a), axis=1)
        return a_cp

    def ber(self, a, b):
        assert a.shape == b.shape
        a_list = a[:, self.carrier].reshape((1, -1))
        b_list = b[:, self.carrier].reshape((1, -1))
        a_bits = self.qpsk_demod(a_list)
        b_bits = self.qpsk_demod(b_list)
        error_count = np.sum(a_bits != b_bits)
        total_bits = a_bits.shape[1]
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
