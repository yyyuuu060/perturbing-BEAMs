import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy.fftpack import fft, fftshift, ifft
import warnings
import pywt
import mne
from sklearn.model_selection import ShuffleSplit, cross_val_score
import os
import math
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
import pandas as pd
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
mne.set_log_level(False)
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
np.random.seed(1234)

bandFreqs = [
    {'name': 'Delta', 'fmin': 0.5, 'fmax': 4},
    {'name': 'Theta', 'fmin': 4, 'fmax': 8},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
    {'name': 'Beta', 'fmin': 13, 'fmax': 30}, ]


def __CalcWP(data, sfreq, wavelet, maxlevel, band):
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    freqBand = (sfreq / 2) / (2 ** maxlevel)
    bandResult = []
    #######################根据实际情况计算频谱对应关系，这里要注意系数的顺序
    for iter_freq in band:
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freq['fmin'] <= bandMin and iter_freq['fmax'] >= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data
        # print([n.path for n in new_wp.get_leaf_nodes(False)])
        # 计算对应频率的数据
        bandResult.append(new_wp.reconstruct(update=True))
    return bandResult, wp


if __name__ == '__main__':
    DA = ['aaaaaaad', 'aaaaaada', 'aaaaaadd', 'aaaaadaa', 'aaaaadad', 'aaaaadda', 'aaaaaddd']
    TA = ['aaaadaaa', 'aaaadaad', 'aaaadada', 'aaaadadd', 'aaaaddaa', 'aaaaddad', 'aaaaddda', 'aaaadddd']
    AA = ['aaadadaa', 'aaadadad', 'aaaddaaa', 'aaaddaad', 'aaaddada', 'aaaddadd', 'aaadddaa', 'aaadddad', 'aaadddda',
          'aaaddddd']
    BA = ['aaadaaaa', 'aaadaaad', 'aaadaada', 'aaadaadd', 'aaadadda', 'aaadaddd', 'aadaadaa', 'aadaadad', 'aadaadda',
          'aadaaddd', 'aadadaaa', 'aadadaad', 'aadadada', 'aadadadd', 'aadaddaa', 'aadaddad', 'aadaddda', 'aadadddd',
          'aaddaaaa', 'aaddaaad', 'aaddaada', 'aaddaadd', 'aaddadaa', 'aaddadad', 'aaddadda', 'aaddaddd', 'aadddaaa',
          'aadddaad', 'aadddada', 'aadddadd','aaddddaa', 'aaddddad', 'aaddddda', 'aadddddd']
    A = [DA, TA, AA, BA]
    test_data = sio.loadmat("test_data.mat")["data"]  # [* ,23,1280]
    attack_data = sio.loadmat("Mix_0.1.mat")["data"]  # [* 5,4,23]
    attack_data = attack_data.swapaxes(1, 3)
    attack_data = attack_data.swapaxes(2, 3)  # [* 22,5,4]
    people = ['AF7', 'FT7', 'T7', 'PO7', 'AF3', 'FC3', 'CP3', 'PO3', 'AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8', 'T8',
              'PO8', 'FCz', 'CPz', 'P7', 'T9', 'Cz', 'T10', 'P8']
    for i in range(attack_data.shape[0]):
        # fig, ax = plt.subplots(nrows=22, ncols=1, figsize=(8, 4), gridspec_kw=dict(top=0.9), sharex=True, sharey=True)
        for s in range(attack_data.shape[1]):
            # ax[s].plot(test_data[i][s][0:256])
            for e in range(attack_data.shape[2]):
                bandResult, wp = __CalcWP(test_data[i][s][0+256*e:256+256*e], 256, wavelet='db1', maxlevel=8, band=bandFreqs)
                for c in range(4):
                    Initial = np.mean(list(map(lambda num:num, abs(fft(bandResult[c])))))
                    attack = attack_data[i][s][e][c]
                    real_f = np.nan_to_num(fft(bandResult[c]).real / abs(fft(bandResult[c]).real))
                    imag_f = np.nan_to_num(fft(bandResult[c]).imag / abs(fft(bandResult[c]).imag))
                    if attack-Initial > 0:
                        real = np.nan_to_num(np.sqrt((attack**2 - Initial**2) * fft(bandResult[c]).real**2 / (
                                    fft(bandResult[c]).real**2 + fft(bandResult[c]).imag**2) + fft(
                            bandResult[c]).real ** 2))
                        imag = np.nan_to_num(np.sqrt((attack**2 - Initial**2) * fft(bandResult[c]).imag**2 / (
                                    fft(bandResult[c]).real**2 + fft(bandResult[c]).imag**2) + fft(
                            bandResult[c]).imag ** 2))
                    else:
                        real = np.nan_to_num(np.sqrt(fft(bandResult[c]).real ** 2 - (attack**2 - Initial**2) * fft(bandResult[c]).real ** 2 / (
                                fft(bandResult[c]).real ** 2 + fft(bandResult[c]).imag ** 2)))
                        imag = np.nan_to_num(np.sqrt(fft(bandResult[c]).imag ** 2 - (attack**2 - Initial**2) * fft(bandResult[c]).imag ** 2 / (
                                fft(bandResult[c]).real ** 2 + fft(bandResult[c]).imag ** 2)))
                    eal = []
                    for v in range(len(real)):
                        eal.append(complex(real_f[v]*real[v], imag_f[v]*imag[v]))
                    data = ifft(eal)
                    new_d = pywt.WaveletPacket(data=data, wavelet='db1', mode='symmetric', maxlevel=8)
                    for f in range(len(A[c])):
                        wp[A[c][f]] = new_d[A[c][f]]
                new_wwp = wp.reconstruct(update=True)
                test_data[i][s][0 + 256 * e:256 + 256 * e] = new_wwp
        #     ax[s].plot(test_data[i][s][0:256])
        #     ax[s].spines['bottom'].set_color('none')
        #     ax[s].spines['left'].set_color('none')
        #     ax[s].spines['top'].set_color('none')
        #     ax[s].spines['right'].set_color('none')
        #     ax[s].yaxis.set_major_locator(plt.NullLocator())
        #     ax[s].xaxis.set_major_formatter(plt.NullFormatter())
        #     ax[s].set_ylabel(people[s])
        # ax[0].legend(['Initial data', 'Perturbation data'], frameon=False, loc=(0.65, 0.6))
        # plt.show()
    sio.savemat('time_Mix_0.1.mat', {'data': test_data})