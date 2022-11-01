import scipy.io as sio
import numpy as np
from matplotlib.pylab import mpl
from scipy.fftpack import fft, fftshift, ifft
import warnings
import pywt
from sklearn.preprocessing import scale
import mne
from sklearn.model_selection import ShuffleSplit
import os
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf

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
    path = r"E:\SeizuresCounter\Epileptic"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    dd = []
    for i, file in enumerate(files):
        raw = read_raw_edf(path + "/" + file, preload=True)
        # print(raw.info.ch_names)
        raw.filter(0.5, 50., fir_design='firwin', skip_by_annotation='edge')
        new_events = mne.make_fixed_length_events(raw, start=0, duration=5, overlap=2.)
        epochs = mne.Epochs(raw, new_events, event_id=None, tmin=0., tmax=5, baseline=None,
                            picks=["FP1-F7", "F7-T7", "T7-P7",
                                   "P7-O1", "FP1-F3", "F3-C3",
                                   "C3-P3", "P3-O1", "FP2-F4",
                                   "F4-C4", "C4-P4", "P4-O2",
                                   "FP2-F8", "F8-T8", "T8-P8-0",
                                   "P8-O2", "FZ-CZ", "CZ-PZ",
                                   "P7-T7", "T7-FT9", "FT10-T8", "T8-P8-1"],
                            preload=True, reject=None, flat=None, proj=True, decim=1, reject_tmin=None,
                            reject_tmax=None,
                            detrend=None,
                            on_missing='raise', reject_by_annotation=True, metadata=None, event_repeated='error',
                            verbose=None)
        data = epochs.get_data()
        data = data * 1000000
        if i == 0:
            dd = data
        else:
            dd = np.concatenate([dd, data], axis=0)
    print(dd.shape)

    path = r'E:\SeizuresCounter\Non_Epileptic'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    dv = []
    for i, file in enumerate(files):
        raw = read_raw_edf(path + "/" + file, preload=True)
        raw.filter(0.5, 50., fir_design='firwin', skip_by_annotation='edge')
        new_events = mne.make_fixed_length_events(raw, start=0, duration=5, overlap=0.)
        epochs = mne.Epochs(raw, new_events, event_id=None, tmin=0., tmax=5, baseline=None,
                            picks=["FP1-F7", "F7-T7", "T7-P7",
                                   "P7-O1", "FP1-F3", "F3-C3",
                                   "C3-P3", "P3-O1", "FP2-F4",
                                   "F4-C4", "C4-P4", "P4-O2",
                                   "FP2-F8", "F8-T8", "T8-P8-0",
                                   "P8-O2", "FZ-CZ", "CZ-PZ",
                                   "P7-T7", "T7-FT9", "FT10-T8", "T8-P8-1"], preload=True,
                            reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None,
                            detrend=None,
                            on_missing='raise', reject_by_annotation=True, metadata=None, event_repeated='error',
                            verbose=None)
        data = epochs.get_data()
        data = data * 1000000
        ccv = ShuffleSplit(10, test_size=0.01, random_state=42)
        # 根据设计的交叉验证参数,分配相关的训练集和测试集数据
        ccv_data = ccv.split(data)
        for train_idx, test_idx in ccv_data:
            x_data, y_data = data[train_idx], data[test_idx]
        if i == 0:
            dv = y_data
        else:
            dv = np.concatenate([dv, y_data], axis=0)
    print(dv.shape)
    data = np.concatenate([dd, dv], axis=0)

    data = np.swapaxes(data, 0, 1)
    for h in range(data.shape[0]):
        data[h] = \
            scale(data[h])
    data = np.swapaxes(data, 0, 1)

    CC = np.zeros([data.shape[0]])
    CC[0:dd.shape[0]] = 1
    CC[dd.shape[0]:data.shape[0]] = 0
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    # 根据设计的交叉验证参数,分配相关的训练集和测试集数据
    cv_data = cv.split(data)
    cv_label = cv.split(CC)
    for train_idx, test_idx in cv_data:
        x_train, x_test = data[train_idx], data[test_idx]
    for train_idx, test_idx in cv_label:
        y_train, y_test = CC[train_idx], CC[test_idx]
    sio.savemat('test_data.mat', {'data': x_test})
    sio.savemat('test_label.mat', {'label': y_test})
    sio.savemat('train_data.mat', {'data': x_train})
    sio.savemat('train_label.mat', {'label': y_train})

    all_train = np.zeros([x_train.shape[0], 5, 4, 22])
    for c in range(x_train.shape[0]):
        for d in range(x_train.shape[1]):
            for x in range(5):
                bandResult, wp = __CalcWP(x_train[c][d][0 + 256 * x:256 + 256 * x], 256, wavelet='db1', maxlevel=8,
                                          band=bandFreqs)
                Delta = abs(fft(bandResult[0]))  # ifft（fft（x））= x
                Theta = abs(fft(bandResult[1]))  # print(fft(bandResult[1]).real)  # 获取实部
                Alpha = abs(fft(bandResult[2]))  # print(fft(bandResult[1].imag)  # 获取虚部
                Beta = abs(fft(bandResult[3]))  # 实部和虚部的平方和等于Beta的平方
                all_train[c][x][0][d] = np.mean(list(map(lambda num:num, Delta)))
                all_train[c][x][1][d] = np.mean(list(map(lambda num:num, Theta)))
                all_train[c][x][2][d] = np.mean(list(map(lambda num:num, Alpha)))
                all_train[c][x][3][d] = np.mean(list(map(lambda num:num, Beta)))

    all_test = np.zeros([x_test.shape[0], 5, 4, 22])
    for c in range(x_test.shape[0]):
        for d in range(x_test.shape[1]):
            for x in range(5):
                bandResult, wp = __CalcWP(x_test[c][d][0 + 256 * x:256 + 256 * x], 256, wavelet='db1', maxlevel=8,
                                          band=bandFreqs)
                Delta = abs(fft(bandResult[0]))
                Theta = abs(fft(bandResult[1]))
                Alpha = abs(fft(bandResult[2]))
                Beta = abs(fft(bandResult[3]))
                all_test[c][x][0][d] = np.mean(list(map(lambda num:num, Delta)))
                all_test[c][x][1][d] = np.mean(list(map(lambda num:num, Theta)))
                all_test[c][x][2][d] = np.mean(list(map(lambda num:num, Alpha)))
                all_test[c][x][3][d] = np.mean(list(map(lambda num:num, Beta)))
    sio.savemat('FFT_test_data.mat', {'data': all_test})

    ch = ['AF7', 'FT7', 'T7', 'PO7', 'AF3', 'FC3', 'CP3', 'PO3', 'AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8', 'T8', 'PO8',
          'FCz', 'CPz', 'P7', 'T9', 'T10', 'P8']
    fake_info = mne.create_info(ch_names=ch, sfreq=256., ch_types='eeg')
    all_x_train = np.zeros([all_train.shape[0], 5, 4, 22, 22])
    for c in range(all_train.shape[0]):
        for d in range(all_train.shape[1]):
            for x in range(all_train.shape[2]):
                data = all_train[c][d][x].reshape(-1, 1)
                montage = make_standard_montage('standard_1020')
                fake_evoked = mne.EvokedArray(data, fake_info)
                fake_evoked.set_montage(montage)
                Zi = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info,
                                          show=False)
                Zi = np.nan_to_num(Zi)
                all_x_train[c][d][x] = Zi
    sio.savemat('topomap_train.mat', {'data': all_x_train})

    all_x_test = np.zeros([all_test.shape[0], 5, 4, 22, 22])
    for c in range(all_test.shape[0]):
        for d in range(all_test.shape[1]):
            for x in range(all_test.shape[2]):
                data = all_test[c][d][x].reshape(-1, 1)
                montage = make_standard_montage('standard_1020')
                fake_evoked = mne.EvokedArray(data, fake_info)
                fake_evoked.set_montage(montage)
                Zi = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info,
                                          show=False)
                Zi = np.nan_to_num(Zi)
                all_x_test[c][d][x] = Zi
    sio.savemat('topomap_test.mat', {'data': all_x_test})
