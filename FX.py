import numpy as np


def float2fx(a, signed=True, frac=15, approximate=np.floor):
    if signed:
        a_sign = np.sign(a)
        fx_disp = np.multiply(np.abs(a), np.power(2, frac))
        fx_disp = approximate(fx_disp)
        fx = np.divide(fx_disp, np.power(2, frac))
        fx = np.multiply(a_sign, fx)
    else:
        fx_disp = np.multiply(a, np.power(2, frac))
        fx_disp = approximate(fx_disp)
        fx = np.divide(fx_disp, np.power(2, frac))
    return fx

#'''
if __name__ == "__main__":
    in1 = -12.523
    print(float2fx(in1))
#'''
