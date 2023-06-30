import numpy as np

from utils import int16_to_bytes, bytes_to_float


def esn_core_config(s, addr, win, wx, wout):
    """ Configure ESN weights
    :param s:  UDP socket
    :param addr: ESN core IP address and port
    :param win: input weights
    :param wx: reservoir weights
    :param wout: output weights
    :return: none
    """

    caddr = (addr[0], addr[1]+1000)  # Config port number is 1000 higher
    header = np.array([32670, 0, 0, 0])
    cmd_bytes = int16_to_bytes(np.concatenate((header, win.flatten(), wout.flatten())))
    s.sendto(cmd_bytes, caddr)

    header = np.array([32670, 0, 128, 0])
    cmd_bytes = int16_to_bytes(np.concatenate((header, wx.flatten())))
    s.sendto(cmd_bytes, caddr)


def esn_core_processing_oneshot(s, addr, inputs):
    """ Send inputs to ESN core on FPGA
    :param s: UDP socket
    :param addr: ESN core IP address
    :param inputs: time domain receiver signals
    :return: time domain prediction
    """

    num_packets = inputs.shape[0]
    outputs = np.zeros((num_packets, inputs.size // (2 * num_packets)))
    for k in range(num_packets):
        input_bytes = int16_to_bytes(inputs[k, :])
        s.sendto(input_bytes, addr)
        data, addr = s.recvfrom(1024)
        outputs[k, :] = bytes_to_float(data)

    outputs = outputs.reshape((outputs.size // 2, 2))
    return outputs


def esn_core_processing(s, addr, inputs, weights):
    """ Send inputs to ESN core on FPGA
    :param s: UDP socket
    :param addr: ESN core IP address and port
    :param inputs: time domain receiver signals
    :param weights: output weights
    :return: time domain prediction
    """

    caddr = (addr[0], addr[1] + 1000)  # Config port number is 1000 higher
    header = np.array([32670, 0, 64, 0])

    num_packets = inputs.shape[0]
    outputs = np.zeros((num_packets, inputs.size // (2 * num_packets)))
    for k in range(num_packets):
        # update output weights
        cmd_bytes = int16_to_bytes(np.concatenate((header, weights[k, :].flatten())))
        s.sendto(cmd_bytes, caddr)
        # ESN core processing
        input_bytes = int16_to_bytes(inputs[k, :].flatten())
        s.sendto(input_bytes, addr)
        data, addr = s.recvfrom(1024)
        outputs[k, :] = bytes_to_float(data)

    outputs = outputs.reshape((outputs.size // 2, 2))
    return outputs
