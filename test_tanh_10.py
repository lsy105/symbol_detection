import numpy as np
from tanh import tanh
import FX as fx
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400


def myavg(a, b):
    return np.mean(np.abs(a - b))

def mymax(a, b):
    return np.amax(np.abs(a - b))

def plot_zoom_in(tanh_lut):
    step = np.power(float(2), -7)

    data_points = int(np.power(2, 8))

    # Zoom-in plot
    plt.figure(dpi=300)
    fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2, figsize=(6, 6))

    #fig.text(0.5, 0.975, 'Absolute Error Before Improvement, Zoomed-In',
    #         horizontalalignment='center',
    #         verticalalignment='top')
    #fig.text(0.5, 0.04, 's', ha='center', va='center')
    #fig.text(0.06, 0.5, 'Absolute Error', ha='center', va='center', rotation='vertical')

    # AX0
    start = step * 5
    end = start + step
    test_inputs_2 = np.linspace(start, end, data_points, endpoint=False)
    outputs_2 = tanh_lut.get_tanh_lut(test_inputs_2)
    outputs_real_2 = np.tanh(test_inputs_2)
    err = outputs_real_2 - outputs_2
    ax0.plot(test_inputs_2, err * 1e6)
    ax0.set_ylim([0, 3])
    # ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # AX1
    start = step * 10
    end = start + step
    test_inputs_2 = np.linspace(start, end, data_points, endpoint=False)
    outputs_2 = tanh_lut.get_tanh_lut(test_inputs_2)
    outputs_real_2 = np.tanh(test_inputs_2)
    err = outputs_real_2 - outputs_2
    ax1.plot(test_inputs_2, err * 1e6)
    ax1.set_ylim([0, 4])
    # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # AX2
    start = step * 50
    end = start + step
    test_inputs_2 = np.linspace(start, end, data_points, endpoint=False)
    outputs_2 = tanh_lut.get_tanh_lut(test_inputs_2)
    outputs_real_2 = np.tanh(test_inputs_2)
    err = outputs_real_2 - outputs_2
    ax2.plot(test_inputs_2, err * 1e6)
    ax2.set_ylim([0, 9])
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # AX3
    start = step * 160
    end = start + step
    test_inputs_2 = np.linspace(start, end, data_points, endpoint=False)
    outputs_2 = tanh_lut.get_tanh_lut(test_inputs_2)
    outputs_real_2 = np.tanh(test_inputs_2)
    err = outputs_real_2 - outputs_2
    ax3.plot(test_inputs_2, err * 1e6)
    ax3.set_ylim([0, 7])
    # ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    plt.title('Absolute Error Before Improvement, Zoomed-In')
    plt.xlabel("s")
    ylabel = 'Absolute Error (' + '$x 10^{6}$' + ')'
    plt.ylabel(ylabel)

    plt.savefig('zoomed_in_abs_err.png', dpi=500, format='png')
    plt.show(dpi=300)




def plot_tanh(tanh_lut):
    input_bit = tanh_lut.input_bit
    slope_fmt = tanh_lut.slope_fmt
    intercept_fmt = tanh_lut.intercept_fmt

    # Zoomed in plots before new intercept
    # plot_zoom_in(tanh_lut)

    data_points = int(np.power(2, (input_bit + 10)))
    inputs = np.linspace(0, 8, data_points, endpoint=False)
    size_0 = int(np.power(2, input_bit))
    size_1 = int(np.power(2, 10))
    inputs_2d = np.reshape(inputs, (size_0, size_1))

    outputs_lut = tanh_lut.get_tanh_lut(inputs_2d)

    # better intercept
    tanh_lut.gen_better_intercept()

    outputs_lut_better = tanh_lut.get_tanh_lut(inputs_2d)
    outputs_real = np.tanh(inputs_2d)

    print('Regularr', 'avg:', myavg(outputs_real, outputs_lut), 'max:', mymax(outputs_real, outputs_lut))
    print('Improved', 'avg:', myavg(outputs_real, outputs_lut_better), 'max:', mymax(outputs_real, outputs_lut_better))


    # Plot Abs Err Before/After
    '''    
    err = outputs_real - outputs_lut
    err_better = outputs_real - outputs_lut_better
    err_abs = np.abs(err)
    err_abs_better = np.abs(err_better)

    relative_err = err_abs_better / (outputs_real + 10e-12)
    err = err.flatten()
    err_abs = err_abs.flatten() * 1e6
    err_better = err_better.flatten()
    err_abs_better = err_abs_better.flatten() * 1e6
    relative_err = relative_err.flatten()

    ylabel = 'Absolute Error (' + '$x 10^{6}$' + ')'
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 4))
    ax0.plot(inputs, err_abs, 'xkcd:azure')
    ax0.set_ylim([0, 12])
    ax0.set(xlabel='s', ylabel=ylabel, title='Before Improvement')

    ax1.plot(inputs, err_abs_better, 'xkcd:aqua green')
    ax1.set_ylim([0, 12])
    ax1.set(xlabel='s', ylabel=ylabel, title='After Improvement')
    plt.show(dpi=400)
    '''


    # Plot Relative Error
    '''
    start_point = int((data_points / 8) * 0.0078125)
    plt.figure()
    plt.plot(inputs[start_point:], relative_err[start_point:])
    plt.title('Relative Error')
    plt.show()
    '''

    # Zoomed in plots after new intercept
    #plot_zoom_in(tanh_lut)




if __name__ == '__main__':
    tanh_lut = tanh(
        input_bit=10,
        dx_bit=10,
        slope_fmt=(8, 8),
        intercept_fmt=(19, 19),
        max=8,
        better_lut=False,
        verbose=False,
        plot=False)

    plot_tanh(tanh_lut)

    # print("Saving COEs...")
    # tanh_lut.save_slope()
    # tanh_lut.save_intercept()

    '''
    # Random inputs for verification
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



