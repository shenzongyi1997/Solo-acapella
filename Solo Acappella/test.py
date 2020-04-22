# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,100,1000)
def cal(x):
    y = []
    for i in range(x.shape[0]):
        temp_i = np.mod(i,200)
        y.append(np.abs(temp_i - 100))
    y = np.asarray(y)
    return y

def differenceFunction(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]

    Fastest implementation. Use the same approach than differenceFunction_scipy.
    This solution is implemented directly with Numpy fft.


    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv



def cumulativeMeanNormalizedDifferenceFunction(df, N):
    """
    Compute cumulative mean normalized difference function (CMND).

    This corresponds to equation (8) in [1]

    :param df: Difference function
    :param N: length of data
    :return: cumulative mean normalized difference function
    :rtype: list
    """

    cmndf = df[1:] * range(1, N) / np.cumsum(df[1:]).astype(float) #scipy method
    return np.insert(cmndf, 0, 1)




y = cal(x)

plt.figure()
plt.plot(x,y)
plt.savefig("input signal.jpg")
df = differenceFunction(y, len(y), 500)
plt.figure()
plt.plot(range(len(df)),df)
plt.savefig("df.jpg")
cmdf = cumulativeMeanNormalizedDifferenceFunction(df, len(df))
plt.figure()
plt.plot(range(len(cmdf)),cmdf)
plt.savefig("cmdf.jpg")

colors = ['b', 'g', 'r', 'c', 'm']
# ax=[]
# plt.figure()
# plt.plot(range(0,400),y[0:400])
# # plt.show()
# plt.savefig("original.jpg")
# for i in range(1,6):
#     # temp_ax = plt.subplot(1,5,i)
#     # ax.append(temp_ax)
#     temp_x = range(int(200 / 5 * i), int(200 / 5 * i + 400))
#     plt.figure()
#     plt.plot(range(len(temp_x)), y[temp_x], colors[i-1])
#     # plt.show()
#     plt.savefig(str(i) +"fifth shift.jpg")

