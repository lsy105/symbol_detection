import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from fpgaESN import esn_core_config, esn_core_processing_oneshot


def check_results(predict, outputs):

    # check output signal
    fig, ax = plt.subplots()
    ax.plot(predict[:, 0])
    ax.plot(outputs[:, 0])
    ax.set(xlabel='Time', ylabel='Signal', title='Python vs. FPGA Outputs')
    ax.set_xlim(1000, 1500)
    plt.show()

    # Check signal constellation
    ry = outputs[:, 0] + 1j * outputs[:, 1]
    ry = np.reshape(ry, (ry.size // 80, 80))  # CP 16, FFT 64
    yf = np.fft.fft(ry[:, 16:]) / np.sqrt(64)

    yy = yf.flatten()
    plt.scatter(yy.real, yy.imag)
    plt.show()

    # check variance
    print("var(predict) = ", np.var(predict.flatten()))
    print("var(outputs) = ", np.var(outputs.flatten()))


def main():
    # ESN core IP address and base port
    addr = ("192.168.20.20", 8001)

    # UDP send and receive sockets
    usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        usock.bind(("", addr[1]))
    except socket.error as msg:
        print(msg)
        sys.exit()

    # Load test data
    test_data = loadmat(file_name="fpga/test_rc_wifi_data.mat")
    w_in = test_data["w_in"]
    w_x = test_data["w_x"]
    w_out = test_data["w_out"]
    inputs = test_data["inputs"]
    predict = test_data["predict"]

    # Configure ESN core weights
    esn_core_config(usock, addr, w_in, w_x, w_out)

    # Prepare input data buffers
    num_packets = 90
    inputs = inputs.reshape((num_packets, inputs.size // num_packets))
    outputs = esn_core_processing_oneshot(usock, addr, inputs)

    # Validate results
    check_results(predict, outputs)

    # savemat(file_name="debug.mat", mdict={"predict": predict, "outputs": outputs})


if __name__ == '__main__':
    main()
