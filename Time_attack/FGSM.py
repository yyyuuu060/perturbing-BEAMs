import numpy as np
from torch.utils.data import DataLoader, random_split
from Utils import *
from Models import *
import os
import pywt
from sklearn.metrics import mean_squared_error
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
torch.manual_seed(1234)
np.random.seed(1234)
HEAD_SIZE_DEFAULT = 0.095  # in [m]
_BORDER_DEFAULT = 'mean'
_EXTRAPOLATE_DEFAULT = 'auto'
GPUID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
torch.backends.cudnn.enabled = False  #
# 忽略了警告错误的输出
warnings.simplefilter("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def convert_time(attack_data, test_data):
    DA = ['aaaaaaad', 'aaaaaada', 'aaaaaadd', 'aaaaadaa', 'aaaaadad', 'aaaaadda', 'aaaaaddd']
    TA = ['aaaadaaa', 'aaaadaad', 'aaaadada', 'aaaadadd', 'aaaaddaa', 'aaaaddad', 'aaaaddda', 'aaaadddd']
    AA = ['aaadadaa', 'aaadadad', 'aaaddaaa', 'aaaddaad', 'aaaddada', 'aaaddadd', 'aaadddaa', 'aaadddad', 'aaadddda',
          'aaaddddd']
    BA = ['aaadaaaa', 'aaadaaad', 'aaadaada', 'aaadaadd', 'aaadadda', 'aaadaddd', 'aadaadaa', 'aadaadad', 'aadaadda',
          'aadaaddd', 'aadadaaa', 'aadadaad', 'aadadada', 'aadadadd', 'aadaddaa', 'aadaddad', 'aadaddda', 'aadadddd',
          'aaddaaaa', 'aaddaaad', 'aaddaada', 'aaddaadd', 'aaddadaa', 'aaddadad', 'aaddadda', 'aaddaddd', 'aadddaaa',
          'aadddaad', 'aadddada', 'aadddadd','aaddddaa', 'aaddddad', 'aaddddda', 'aadddddd']
    A = [DA, TA, AA, BA]
    # 16，5，22，4
    # 16，5，22，256
    for i in range(attack_data.shape[0]):
        for s in range(attack_data.shape[1]):
            for e in range(attack_data.shape[2]):
                bandResult, wp = __CalcWP(test_data[i][s][e], 256, wavelet='db1', maxlevel=8, band=bandFreqs)
                for c in range(4):
                    Initial = np.mean(list(map(lambda num:num, abs(fft(bandResult[c])))))
                    attack = attack_data[i][s][e][c]
                    real_f = np.nan_to_num(fft(bandResult[c]).real / abs(fft(bandResult[c]).real))
                    imag_f = np.nan_to_num(fft(bandResult[c]).imag / abs(fft(bandResult[c]).imag))
                    if attack**2-Initial**2 > 0:
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
                test_data[i][s][e] = new_wwp
    return test_data

# 16，5，4，22     16，5，1，22，256
def convert(test, images):
    test = np.swapaxes(test, 2, 3)  # 16，5，22，4
    image = np.zeros([images.shape[0], images.shape[1], images.shape[3], images.shape[4]])
    for c in range(image.shape[0]):
        for d in range(image.shape[1]):
            for x in range(image.shape[2]):
                image[c][d][x] = images[c][d][0][x]
    for c in range(test.shape[0]):
        for d in range(test.shape[1]):
            for x in range(test.shape[2]):
                bandResult, wp = __CalcWP(image[c][d][x], 256, wavelet='db1', maxlevel=8, band=bandFreqs)
                Delta = np.mean(list(map(lambda num:num, abs(fft(bandResult[0])))))
                Theta = np.mean(list(map(lambda num:num, abs(fft(bandResult[1])))))
                Alpha = np.mean(list(map(lambda num:num, abs(fft(bandResult[2])))))
                Beta = np.mean(list(map(lambda num:num, abs(fft(bandResult[3])))))
                if test[c][d][x][0] > Delta:
                    test[c][d][x][0] = Delta + (test[c][d][x][0] - Delta)
                if test[c][d][x][0] < Delta:
                    test[c][d][x][0] = Delta - (Delta - test[c][d][x][0])
                if test[c][d][x][1] > Theta:
                    test[c][d][x][1] = Theta + (test[c][d][x][1] - Theta)
                if test[c][d][x][1] < Theta:
                    test[c][d][x][1] = Theta - (Theta - test[c][d][x][1])
                if test[c][d][x][2] > Alpha:
                    test[c][d][x][2] = Alpha + (test[c][d][x][2] - Alpha)
                if test[c][d][x][2] < Alpha:
                    test[c][d][x][2] = Alpha - (Alpha - test[c][d][x][2])
                if test[c][d][x][3] > Delta:
                    test[c][d][x][3] = Beta + (test[c][d][x][3] - Beta)
                if test[c][d][x][3] < Delta:
                    test[c][d][x][3] = Beta - (Beta - test[c][d][x][3])
    attack_images = convert_time(test, image)
    attack_images = attack_images[:, :, np.newaxis, :, :]
    return attack_images


def fgsm_attack(model, loss, images, labels, eps):
    labels = labels.long()
    images.requires_grad = True
    outputs = model(images.cuda())
    model.zero_grad()
    cost = loss(outputs, labels.cuda())
    cost.backward()
    eps = eps * images.grad.sign()
    attack_images = torch.tensor(images+eps, dtype=torch.float32)
    return attack_images


def eval_adv_test_FGSM(model, device, test_loader, x_test):
    """
    evaluate model by white-box attack
    """
    model.eval()
    loss = nn.CrossEntropyLoss()
    correct = 0
    X_correct = 0
    total = 0
    i = 0
    eps = 0.022
    dd = []
    SUM_1 = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        y = torch.tensor(target, dtype=torch.float32)
        X = torch.tensor(data, dtype=torch.float32)
        X_outputs = model(X.cuda())
        _, X_predicted = torch.max(X_outputs, dim=1)
        attack_images = fgsm_attack(model, loss, X, y, eps)
        attack_images = convert(x_test[0 + i * 16:16 + i * 16], attack_images.cpu().numpy())  # 16，5，4，22     16，5，1，22，256
        attack_images = torch.from_numpy(attack_images).to(device)
        # attack_images = attack_images.to(device)
        attack_images = attack_images.to(torch.float32)
        outputs = model(attack_images)
        _, predicted = torch.max(outputs, dim=1)
        total += y.size(0)  # labels 的长度
        y = y.long()
        correct += (predicted == y.cuda()).sum().item()  # 预测正确的数目
        X_correct += (predicted != X_predicted.cuda()).sum().item()  # 攻击成功的数目
        SUM = 0
        o = 0
        for v in range(X.shape[0]):
            for t in range(X.shape[1]):
                for b in range(X.shape[2]):
                    RMSE = mean_squared_error(X[v][t][b].cpu().detach().numpy().flatten(),
                                              attack_images[v][t][b].cpu().detach().numpy().flatten()) ** 0.5
                    SUM = SUM + RMSE
                    o = o + 1
        SUM_1 = SUM_1 + SUM / o
        if i == 0:
            dd = attack_images.cpu()
        else:
            dd = np.concatenate([dd, attack_images.cpu()], axis=0)
        i = i + 1
    print(SUM_1/i)
    print('Accuracy of the network on the  test images: %d %%' % (
            100 * correct / total))
    print('Success rate of the network on the  test images: %d %%' % (
            100 * X_correct / total))
    # sio.savemat('attack_MaxCNN_0.3.mat', {'data': dd})


test_Label = sio.loadmat("test_label.mat")["label"][0]
attack_time_data = sio.loadmat("test_data.mat")["data"]
test_Label = torch.tensor(test_Label, dtype=torch.long)

attack_nn = np.zeros([attack_time_data.shape[0], 5, 22, 256])
for i in range(attack_time_data.shape[0]):
    for r in range(attack_time_data.shape[1]):
        for e in range(5):
            attack_nn[i][e][r] = attack_time_data[i][r][0 + 256 * e:256 + 256 * e]
attack_nn = attack_nn[:, :, np.newaxis, :, :]  #

attack_data = EEGImagesDataset(test_Label, attack_nn)
attack_loader = torch.utils.data.DataLoader(attack_data, batch_size=16, shuffle=False, num_workers=0)
x_test = sio.loadmat("FFT_MaxCNN_0.3.mat")["data"]
XY = pd.read_csv("XY.csv")
XY = np.array(XY)
pos = pd.read_csv("pos.csv")
pos = np.array(pos)
"""
   MaxCNN_best_model.pth
   TempCNN_best_model.pth
   LSTM_best_model.pth
   Mix_best_model.pth
"""
if __name__ == '__main__':
    PATH = r'D:\SeizuresCounter\Time_attack\TempCNN_model.pth'
    test_net = TempCNN().cuda()
    test_net.load_state_dict(torch.load(PATH))
    eval_adv_test_FGSM(test_net, device, attack_loader, x_test)
