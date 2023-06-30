SNR_list = [1000]
for SNR in SNR_list:
    print(SNR)


for i in range(packet):
    RC.run():
        BER = self.train_and_test_inv_RLS(self.rx_time_cp, tx_label_time_CP);
            def train_and_test_inv_RLS(self, rx_time_cp, tx_label_time_cp, train_with_CP=True):
                # verify W_out with training data
                verify_predict_time = self.esn.test_RLS(esn_input_RLS, W_out, continuation=False)

                # test ESN with RLS on data symbols
                predicts_time = self.esn.test_RLS(test_input_RLS, W_out, continuation=True)  # (N_symbols * (N_fft+N_cp), 2)