import numpy as np
from struct import pack

def int16_to_bytes_v3(inputs):
    """Convert 16-bit inputs to byte stream
    :param inputs: numpy arrays (int16 format)
    :return: byte streams (little endian format)
    """

    # convert to unsigned integers
    tmp_idx = np.nonzero(inputs < 0)
    inputs[tmp_idx] = inputs[tmp_idx] + 2**16

    # convert to bytes
    inputs = np.reshape(inputs, (inputs.size, 1))
    inputs = np.concatenate((inputs % (2**8), np.floor(inputs/(2**8))), axis=1)
    inputs = np.reshape(inputs, inputs.size)

    return bytes(inputs.astype('uint8'))


def int16_to_bytes(inputs):
    """Convert 16-bit inputs to byte stream
    :param inputs: numpy arrays (int16 format)
    :return: byte streams (little endian format)
    """

    # convert to unsigned integers
    tmp_idx = np.nonzero(inputs < 0)
    inputs[tmp_idx] = inputs[tmp_idx] + 2**16

    inputs = np.reshape(inputs.astype('int'), (inputs.size, 1))
    inputs = np.concatenate((np.bitwise_and(inputs, 255), np.bitwise_and(np.right_shift(inputs, 8), 255)), axis=1)
    inputs = np.reshape(inputs, inputs.size)

    # convert to bytes
    outputs = bytearray(inputs.tolist())

    return outputs


def bytes_to_float(din):
    """Convert byte stream to float numbers using little endian (LSB first)
    """

    # Convert bytes to int32
    val = np.frombuffer(din, dtype='int32')
    # Using Q24 format
    val = np.true_divide(val, 2**24)

    return val
