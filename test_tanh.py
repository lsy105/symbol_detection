import numpy as np
from tanh import tanh
import FX as fx
import matplotlib.pyplot as plt
from scipy.io import savemat

def plot_tanh(tanh_lut):

    data_points = int(np.power(2, 17))
    inputs = np.linspace(0, 8, data_points, endpoint=False)
    size_0 = int(np.power(2, 8))
    size_1 = int(np.power(2, 9))
    inputs_2d = np.resize(inputs, (size_0, size_1))

    outputs_lut = np.zeros((size_0, size_1))
    '''
    for i in range(0, size_0):
        outputs_lut[i, :] = tanh_lut.get_tanh_lut(inputs_2d[i, :])

    outputs_real = np.tanh(inputs_2d)
    err = np.abs(outputs_real - outputs_lut)
    err_max = np.amin(err, axis=1)
    err = err.flatten()

    # plt.figure(1)
    # plt.plot(inputs, outputs_lut)
    # plt.plot(inputs, outputs_real)
    # plt.show()


    plt.figure(1)
    plt.plot(inputs, err)
    # plt.plot(inputs, err_max)
    plt.show()
    '''
    # better intercept
    tanh_lut.gen_better_intercept()
    for i in range(0, size_0):
        outputs_lut[i, :] = tanh_lut.get_tanh_lut(inputs_2d[i, :])
    outputs_real = np.tanh(inputs_2d)
    err = np.abs(outputs_real - outputs_lut)
    relative_err = err / (outputs_real + 10e-12)
    err_max = np.amin(err, axis=1)
    err = err.flatten()
    relative_err = relative_err.flatten()

    plt.figure(2)
    plt.plot(inputs, err)
    plt.title('Error')
    # plt.plot(inputs, err_max)
    plt.show()

    start_point = int((data_points / 8) * 0.01)
    plt.figure()
    plt.plot(inputs[start_point:], relative_err[start_point:])
    plt.title('Relative Error')
    plt.show()

    # Zoom-in plot
    data_points = int(np.power(2, 9))
    test_inputs_2 = np.linspace(1, 1.032, data_points, endpoint=False)
    outputs_2 = tanh_lut.get_tanh_lut(test_inputs_2)
    outputs_real_2 = np.tanh(test_inputs_2)
    err = outputs_real_2 - outputs_2
    plt.figure(3)
    plt.plot(test_inputs_2, err)
    plt.ylabel('Zoom-in')
    plt.show()




if __name__ == '__main__':
    tanh_lut = tanh(input_bit=8, slope_fmt=(10, 10), intercept_fmt=(19, 19), verbose=False)
    
    '''
    np.random.seed(123)
    test_inputs = -8 + (8 - (-8)) * np.random.random_sample(size=(20000,))
    test_outputs = tanh_lut.get_tanh_lut(fx.float2fx(test_inputs, frac=30))
    tanh_real = np.tanh(test_inputs)

    err = test_outputs - tanh_real
    print("test_inputs", test_inputs)
    print("test_outputs", test_outputs)
    print("tanh_real", tanh_real)
    print("err:", err)

    err_avg = np.sum(np.abs(err)) / err.size
    err_max = np.max(np.abs(err))
    err_min = np.min(np.abs(err))
    print("err_avg", err_avg, "max", err_max, "min", err_min)

    test_i = np.array([0, 0.0625, 0.125, 0.002, 0.07])
    test_o = tanh_lut.get_tanh_lut(fx.float2fx(test_i, frac=30))
    '''

    plot_tanh(tanh_lut)
    print("Saving slopes in COE...")
    tanh_lut.save_slope()
    tanh_lut.save_intercept()
    
    savemat(file_name='tanh_lut.mat', mdict={'tanh_intercept': tanh_lut.intercept, 
		'tanh_slope': tanh_lut.slope});

    # print(np.log2(1-9.99023438e-01))
