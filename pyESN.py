import numpy as np
import copy
from scipy.io import savemat

import FX as fx
from fpgaESN import esn_core_config, esn_core_processing

def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, silent=True, lut_activation=False, tanh_lut=None):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builtin RandomState.
            silent: suppress messages
        """
        # check for proper dimensionality of all arguments and write them down.
        self.RLS_lambda = 0.9985
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        # for RLS algorithm
        self.psi = np.identity(self.n_inputs + self.n_reservoir) * 1
        self.psi_inv = np.linalg.inv(self.psi)
        self.W_out = np.zeros((self.n_outputs, self.n_inputs + self.n_reservoir))

        self.train_ini_state = np.zeros(self.n_reservoir)
        self.train_ini_teacher = np.zeros(self.n_outputs)

        self.test_ini_state = np.zeros(self.n_reservoir)
        self.test_ini_teacher = np.zeros(self.n_outputs)

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent

        # For LUT implementation of Tanh
        self.lut_activation = lut_activation
        self.tanh_lut = tanh_lut

        self.initweights()

    def initweights(self):
        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity):
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(
            self.n_reservoir, self.n_inputs) * 2 - 1
        # random feedback (teacher forcing) weights:
        self.W_feedb = self.random_state_.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern))
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))

        '''
        return (np.tanh(preactivation)
                + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))
        '''
        if self.lut_activation:
            return (self.tanh_lut.get_tanh_lut(preactivation) +
                    self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))
        else:
            return (np.tanh(preactivation)
                    + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def _update_np(self, state, input_pattern, output_pattern):
        preactivation = (np.dot(self.W, state)
                         + np.dot(self.W_in, input_pattern))
        return (np.tanh(preactivation)
                + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def _update_lut(self, state, input_pattern, output_pattern):
        preactivation = (np.dot(self.W, state)
                         + np.dot(self.W_in, input_pattern))
        return (self.tanh_lut.get_tanh_lut(preactivation) +
                self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def _update_inv(self, state, input_pattern, output_pattern):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern))
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))

        if self.lut_activation:
            return (self.tanh_lut.get_tanh_lut(preactivation) +
                    self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))
        else:
            return (np.tanh(preactivation)
                    + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, inspect=False):
        """
        Collect the network's reaction to training data, train readout weights.

        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states

        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if not self.silent:
            print("harvesting states...")
        # step the reservoir through the given input,output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],
                                        teachers_scaled[n - 1, :])

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
        # we'll disregard the first few states:
        transient = min(int(inputs.shape[1] / 10), 100)
        # include the raw inputs:
        extended_states = np.hstack((states, inputs_scaled))
        # Solve for W_out:
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                            self.inverse_out_activation(teachers_scaled[transient:, :])).T

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:
            from matplotlib import pyplot as plt
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto',
                       interpolation='nearest')
            plt.colorbar()

        if not self.silent:
            print("training error:")
        # apply learned weights to the collected states:
        pred_train = self._unscale_teacher(self.out_activation(
            np.dot(extended_states, self.W_out.T)))
        if not self.silent:
            print(np.sqrt(np.mean((pred_train - outputs) ** 2)))
        return pred_train

    def predict(self, inputs, continuation=True):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state

        Returns:
            Array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out,
                                                           np.concatenate([states[n + 1, :], inputs[n + 1, :]])))

        return self._unscale_teacher(self.out_activation(outputs[1:]))

    def RSL_psi_inv(self, extended_state, psi_inv_pre):
        lambda_temp = self.RLS_lambda
        # lambda_temp=1
        extended_state = extended_state.reshape((-1, 1))
        u = np.matmul(psi_inv_pre, extended_state)  # (N_reservoir + N_in, 1)
        k = float(1) / (lambda_temp + np.matmul(extended_state.T, u)) * u  # (N_reservoir + N_in, 1)
        psi_inv_current = float(1) / lambda_temp * (
                psi_inv_pre - np.matmul(k, np.matmul(extended_state.T, psi_inv_pre)))
        return psi_inv_current

    def RLS_single_step(self, extended_state, teacher_scaled, W_out_pre, psi_inv_pre):
        """
        Update W_out single step

        Args:
            extended_state:  (N_reservoir + N_in, 1)
            teacher_scaled: (N_out, 1)
            W_out_pre:  (N_out, N_reservoir + N_in)
            psi_inv_pre: (N_reservoir + N_in, N_reservoir + N_in)

        Returns:
            W_out_current: (N_out, N_reservoir + N_in)
            psi_inv_current: (N_reservoir + N_in, N_reservoir + N_in)
        """
        extended_state = extended_state.reshape((-1, 1))
        teacher_scaled = teacher_scaled.reshape((-1, 1))
        u = np.matmul(psi_inv_pre, extended_state)  # (N_reservoir + N_in, 1)
        k = 1 / (self.RLS_lambda + np.matmul(extended_state.T, u)) * u  # (N_reservoir + N_in, 1)
        y = np.matmul(W_out_pre, extended_state)  # (N_out, 1)
        e = teacher_scaled - y  # (N_out, 1)
        W_out_current = W_out_pre + np.matmul(e, k.T)  # (N_out, N_reservoir + N_in)
        psi_inv_current = float(1) / self.RLS_lambda * (
                psi_inv_pre - np.matmul(k, np.matmul(extended_state.T, psi_inv_pre)))

        return W_out_current, psi_inv_current, y.squeeze(), e.squeeze()

    def train_RLS(self, inputs, teachers, continuation=False):
        """
        :param continuation: Use last state and label from previous training for initial state and label for current training
        :param inputs: (N_samples, N_in)
        :param teachers: (N_samples, N_out)
        :return: W_out_set: (N_samples, N_out, N_in + N_reservoir)
                error_unscaled: (N_samples, N_out)
        """

        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(teachers)

        savemat(file_name='fpga/test_rc_wifi_train.mat',
                mdict={"train_inputs": inputs_scaled, "train_label": teachers_scaled})
        
        teachers_scaled_pre = np.concatenate((self.train_ini_teacher[np.newaxis, :], teachers_scaled),
                                             axis=0)  # (N_samples + 1, N_out)

        state_pre = copy.deepcopy(self.train_ini_state)

        W_out_set = np.zeros((inputs.shape[0], self.n_outputs, self.n_inputs + self.n_reservoir))
        error_unscaled = 100 * np.ones((inputs.shape[0], self.n_outputs))
        predicts = np.zeros((inputs.shape[0], self.n_outputs))
        error_unscaled_pre_w = 100 * np.ones((inputs.shape[0], self.n_outputs))
        predicts_pre_w = np.zeros((inputs.shape[0], self.n_outputs))
        
        
        for n in range(inputs.shape[0]):
            # Examine the impact of tanh_lut
            '''
            state_nptanh = self._update_np(state_pre, inputs_scaled[n, :], teachers_scaled_pre[n,:])
            state_luttanh = self._update_lut(state_pre, inputs_scaled[n, :], teachers_scaled_pre[n, :])
            state_diff = state_luttanh - state_nptanh
            rmse = np.sqrt(state_diff.dot(state_diff) / float(state_diff.size))
            print('state_diff', 'max:', np.amax(state_diff), 'min:', np.amin(state_diff), 'rmse:', rmse)
            '''

            state = self._update(state_pre, inputs_scaled[n, :], teachers_scaled_pre[n, :])
            extended_state = np.concatenate([state, inputs_scaled[n, :]])
            self.W_out, self.psi_inv, predicts_pre_w[n, :], error_unscaled_pre_w[n, :] = self.RLS_single_step(
                extended_state, teachers_scaled[n], self.W_out, self.psi_inv)
            predicts[n, :] = self._unscale_teacher(self.out_activation(np.dot(self.W_out, extended_state)))
            error_unscaled[n, :] = predicts[n, :] - teachers[n, :]
            
            W_out_set[n, :] = self.W_out
            state_pre = copy.deepcopy(state)

        # print('Average training error', np.mean(np.abs(error_unscaled)))
        # print('Average training error using previous w', np.mean(np.abs(error_unscaled_pre_w)))
        # Save initial state and output for next training batch
        if continuation:
            self.train_ini_state = copy.deepcopy(state)
            self.train_ini_teacher = copy.deepcopy(teachers_scaled[-1, :])
        else:
            pass
            # self.train_ini_state = np.zeros(self.n_reservoir)
            # self.train_ini_teacher = np.zeros(self.n_outputs)
            # self.W_out = np.zeros((self.n_outputs, self.n_inputs + self.n_reservoir))

        return W_out_set, error_unscaled, predicts, error_unscaled_pre_w, predicts_pre_w

    def test_RLS(self, inputs, W_out_set, i_packet, continuation=False, debug = False, output_folder = 'data_outputs/S2'):
        #print('debug:', debug, '\noutput_folder:', output_folder)

        if W_out_set.ndim < 3:
            print('Using same W_out for all samples')
            W_out_repeat = np.zeros((inputs.shape[0], W_out_set.shape[0], W_out_set.shape[1]))
            W_out_repeat[:, :] = W_out_set
            W_out_set = W_out_repeat
        inputs_scaled = self._scale_inputs(inputs)

        # if debug:
        #     print('i=', i_packet, 'in test_RLS')
        # print('in test_RLS')

        state_pre = copy.deepcopy(self.test_ini_state)
        teacher_pre = copy.deepcopy(self.test_ini_teacher)
        predict_set = np.zeros((inputs.shape[0], self.n_outputs))
        state_set = np.zeros((inputs.shape[0], self.n_reservoir + self.n_inputs))
        for n in range(inputs.shape[0]):
            state = self._update(state_pre, inputs_scaled[n, :], teacher_pre)
            extended_state = np.concatenate([state, inputs_scaled[n, :]])
            state_set[n, :] = extended_state
            predict = self.out_activation(np.dot(W_out_set[n, :], extended_state))
            predict_unscaled = self._unscale_teacher(predict)
            import torch
            diff_scale = torch.nn.MSELoss()(torch.tensor(predict), torch.tensor(predict_unscaled))
           
            #print('predict_unscaled:', predict_unscaled)
            predict_set[n, :] = predict_unscaled
            teacher_pre = copy.deepcopy(predict)
            state_pre = copy.deepcopy(state)

            # save the Wout for each symbol
            wout_update_cycle = 100
            set_num = int (np.floor (n / wout_update_cycle) * wout_update_cycle)
            if n == set_num:
                W_out_tosave = W_out_set[n, :] * 1
                #W_out_tosave[: ,-4:] = W_out_tosave[: ,-4:] / 4
                W_out_tosave_temp = fx.float2fx(W_out_tosave, frac=5) * np.power(int(2), 5)  # 10Q5
                wout_fname = output_folder + '/' + str(i_packet) + '/w_out_' + str(n) + '.dat'
                if debug:
                    print('saving W_out #' + str(n) + '...')
                np.savetxt(wout_fname, W_out_tosave_temp.flatten().astype(int), delimiter='\n', fmt='%d')

        
        #print('predict_set:', predict_set)
        #print('state_set:', state_set)
        #"""
        # save test vectors for rtl simulation
        # inputs_tosave = fx.float2fx(inputs * 64, frac=15) * np.power(int(2), 15)
        inputs_tosave = fx.float2fx(inputs_scaled, frac=15) * np.power(int(2), 15) # Q15
        # W_in_tosave = fx.float2fx(self.W_in * np.power(float(2), -6), frac=15) * np.power(int(2), 15)
        W_in_tosave = fx.float2fx(self.W_in, frac=15) * np.power(int(2), 15) # Q15
        W_x_tosave = fx.float2fx(self.W, frac=15) * np.power(int(2), 15) #Q15
        state_tosave = fx.float2fx(state_set, frac=19) * np.power(int(2), 19)  # Q19
        W_out_tosave = W_out_set[0, :]
        # W_out_tosave[:, -4:] = W_out_tosave[:, -4:] * np.power(float(2), -6)
        #W_out_tosave = fx.float2fx(W_out_tosave, frac=5) * np.power(int(2), 5) # 10Q5
        predict_tosave = predict_set
        #print('predict_tosave:', predict_tosave)


        #FPGA_folder_name = 'FPGA_new/LOS_Near/packet' + str(i_packet)
        FPGA_folder_name = output_folder + '/' + str(i_packet)


        np.savetxt(FPGA_folder_name + '/inputs.dat', inputs_tosave.flatten().astype(int), delimiter='\n', fmt='%d')
        np.savetxt(FPGA_folder_name + '/w_in.dat', W_in_tosave.flatten().astype(int), delimiter='\n', fmt='%d')
        np.savetxt(FPGA_folder_name + '/w_x.dat', W_x_tosave.flatten().astype(int), delimiter='\n', fmt='%d')
        np.savetxt(FPGA_folder_name + '/state_tosave.dat', state_tosave.flatten().astype(int), delimiter='\n', fmt='%d')
        #np.savetxt(FPGA_folder_name + '/w_out.dat', W_out_tosave.flatten().astype(int), delimiter='\n', fmt='%d')
        #np.savetxt(FPGA_folder_name + '/predict_tosave.dat', predict_tosave.flatten(), delimiter='\n')
        np.savetxt(FPGA_folder_name + '/predict_tosave.dat', predict_tosave, delimiter='\n')

        savemat(file_name=FPGA_folder_name + '/test_rc_wifi_float.mat',
                mdict={"inputs": inputs_scaled, "w_in": self.W_in, "w_x": self.W,
                       "w_out": W_out_set[0, :], "predict": predict_tosave, "init_state": self.test_ini_state})

        savemat(file_name=FPGA_folder_name + '/test_rc_wifi_data.mat',
                mdict={"inputs": inputs_tosave, "w_in": W_in_tosave, "w_x": W_x_tosave,
                       "w_out": W_out_tosave, "predict": predict_tosave, "init_state": self.test_ini_state})
        #"""


        # Save initial state and output for next testing batch
        if continuation:
            self.test_ini_state = copy.deepcopy(state)
            self.test_ini_teacher = copy.deepcopy(predict)
        else:
            pass
            # self.test_ini_state = np.zeros(self.n_reservoir)
            # self.test_ini_teacher = np.zeros(self.n_outputs)

        return predict_set

    def test_RLS_FPGA(self, usock, addr, inputs, W_out_set, continuation=False):
        if W_out_set.ndim < 3:
            print('Using same W_out for all samples')
            W_out_repeat = np.zeros((inputs.shape[0], W_out_set.shape[0], W_out_set.shape[1]))
            W_out_repeat[:, :] = W_out_set
            W_out_set = W_out_repeat
        inputs_scaled = self._scale_inputs(inputs)

        # convert to fixed-point numbers for ESN-FPGA core processing
        w_in_fpga = np.floor(self.W_in * (2**15))  # Q15
        w_x_fpga = np.floor(self.W * (2**15))  # Q15
        w_out_fpga = np.floor(W_out_set[::80, :] * (2**5))  # 10Q5
        inputs_fpga = np.floor(inputs_scaled * (2 ** 15))  # Q15

        esn_core_config(usock, addr, w_in_fpga, w_x_fpga, w_out_fpga[0, :])

        num_packets = inputs_fpga.shape[0] // 80  # OFDM symbol length is 80
        inputs_fpga = inputs_fpga.reshape((num_packets, inputs_fpga.size // num_packets))
        predict_set = esn_core_processing(usock, addr, inputs_fpga,  w_out_fpga)

        """
        # Save initial state and output for next testing batch
        if continuation:
            self.test_ini_state = copy.deepcopy(state)
            self.test_ini_teacher = copy.deepcopy(predict)
        else:
            pass
        """

        return predict_set


    def train_inv(self, inputs, teachers, continuation=False):
        """
                :param continuation: Use last state and label from previous training for initial state and label for current training
                :param inputs: (N_samples, N_in)
                :param teachers: (N_samples, N_out)
                :return: W_out_set: (N_samples, N_out, N_in + N_reservoir)
                        error_unscaled: (N_samples, N_out)
                """

        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(teachers)

        teachers_scaled_pre = np.concatenate((self.train_ini_teacher[np.newaxis, :], teachers_scaled),
                                             axis=0)  # (N_samples + 1, N_out)
        state_pre = copy.deepcopy(self.train_ini_state)
        state = np.zeros((inputs.shape[0], self.n_reservoir))

        for n in range(inputs.shape[0]):
            state[n, :] = self._update(state_pre, inputs_scaled[n, :], teachers_scaled_pre[n, :])
            state_pre = copy.deepcopy(state[n, :])
        extended_states = np.concatenate((state, inputs_scaled), axis=1)
        # transient = min(int(inputs.shape[1] / 10), 100)
        transient = 16
        try:
            self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                                self.inverse_out_activation(teachers_scaled[transient:, :])).T
        except np.linalg.LinAlgError as e:
            print(e)

        # print("training error:")
        # apply learned weights to the collected states:
        pred_train = self._unscale_teacher(self.out_activation(np.dot(extended_states, self.W_out.T)))
        # print(np.sqrt(np.mean((pred_train - teachers)**2)))
        # Save initial state and output for next training batch
        if continuation:
            self.train_ini_state = copy.deepcopy(state[-1, :])
            self.train_ini_teacher = copy.deepcopy(teachers_scaled[-1, :])
            for i in range(extended_states.shape[0]):
                self.psi_inv = self.RSL_psi_inv(extended_states[i, :], self.psi_inv)
            # self.psi_inv = np.identity(extended_states.shape[1]) * 13
            # N_samples = extended_states.shape[0]
            # lambda_matrix = np.identity(N_samples)
            # for i in range(N_samples):
            #     lambda_matrix[i, i] = self.RLS_lambda ** (N_samples - i - 1)
            # self.psi_inv = np.linalg.inv(np.matmul(np.matmul(extended_states.T, lambda_matrix), extended_states))
        else:
            pass
            # self.train_ini_state = np.zeros(self.n_reservoir)
            # self.train_ini_teacher = np.zeros(self.n_outputs)
            # self.W_out = np.zeros((self.n_outputs, self.n_inputs + self.n_reservoir))

        return pred_train

    def test_inv(self, inputs, continuation=False):
        inputs_scaled = self._scale_inputs(inputs)

        state_pre = copy.deepcopy(self.test_ini_state)
        teacher_pre = copy.deepcopy(self.test_ini_teacher)
        predict_set = np.zeros((inputs.shape[0], self.n_outputs))
        for n in range(inputs.shape[0]):
            state = self._update(state_pre, inputs_scaled[n, :], teacher_pre)
            extended_state = np.concatenate([state, inputs_scaled[n, :]]) # FIXME Bug found!
            predict = self.out_activation(np.dot(self.W_out, extended_state))
            predict_unscaled = self._unscale_teacher(predict)
            predict_set[n, :] = predict_unscaled
            teacher_pre = copy.deepcopy(predict)
            state_pre = copy.deepcopy(state)

        # Save initial state and output for next testing batch
        if continuation:
            self.test_ini_state = copy.deepcopy(state)
            self.test_ini_teacher = copy.deepcopy(predict)
        else:
            pass
            # self.test_ini_state = np.zeros(self.n_reservoir)
            # self.test_ini_teacher = np.zeros(self.n_outputs)

        return predict_set


#add from Victor_original

    def predict_hw(self, inputs, W_out_set, continuation=False, debug=False, debug_lv=10, wout_update_cycle=100):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state

        Returns:
            Array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])

        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        #inputs = fx.float2fx(inputs * 8, frac=28)
        # inputs = inputs * 8
        inputs = fx.float2fx(inputs * 4, frac=15)

        for n in range(n_samples):
            if debug and n < debug_lv:
                debug2 = True
            else:
                debug2 = False
            if debug and n < debug_lv:
                print("-----------------------------------")
                print("loop", n)
            states[
                n + 1, :] = self._update_hw(state=states[n, :], input_pattern=inputs[n + 1, :], debug=debug2)
            if debug and n < debug_lv:
                print("laststates", states[n, :])
                print("echostates", states[n + 1, :])
                print("input", inputs[n + 1, :])

            set_num = int(np.floor(n / wout_update_cycle) * wout_update_cycle)
            W_out_hw = W_out_set[set_num, :] * 1
            W_out_hw[:, -4:] = W_out_hw[:, -4:] / 4
            W_out_hw = fx.float2fx(W_out_hw, frac=-3)

            outputs[n + 1, :] = fx.float2fx(np.dot(W_out_hw, np.concatenate([states[n + 1, :], inputs[n + 1, :]])), frac=12)
            if debug and n < debug_lv:
                print("W_out_fx", W_out_hw, W_out_hw.shape)
                print("extended_state", np.concatenate([states[n + 1, :], inputs[n + 1, :]]), np.concatenate([states[n + 1, :], inputs[n + 1, :]]).shape)
                print("output", outputs[n + 1, :])
                print("-----------------------------------")
            if debug and (n==2000 or n==4000 or n==7000):
                print('time step', n)
                print("W_out_fx", W_out_set[n, :], W_out_set[n, :].shape)

        return self._unscale_teacher(outputs[1:])


    def _update_hw(self, state, input_pattern, debug=False):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        W = fx.float2fx(self.W, frac=15)
        W_in = fx.float2fx(self.W_in / 4, frac=15)
        #W = self.W
        #W_in = self.W_in


        preactivation = (np.dot(W, state)
                         + np.dot(W_in, input_pattern))

        preactivation_fx = fx.float2fx(preactivation, frac=40)
        if debug:
            print("sop_fx", preactivation_fx)

        # return np.tanh(preactivation_fx)
        # return np.tanh(preactivation)
        return self.tanh_lut.get_tanh_lut(preactivation)
