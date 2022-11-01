import torch.backends.cudnn
from Utils import *
from Models import *
import warnings
import argparse
import pandas as pd
from scipy import interpolate
import mne
from mne.channels import make_standard_montage
import numpy as np
from scipy.optimize import differential_evolution
import torch.nn as nn
from torch.autograd import Variable
import os
from sklearn.metrics import mean_squared_error
torch.backends.cudnn.enabled = False
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser(description='PyTorch MNIST PGD Attack Evaluation')
parser.add_argument('--gpuid', default=0,
                    help='gpuid')
parser.add_argument('--d', type=int, default=20, help='number of pixels to change')
parser.add_argument('--iters', type=int, default=30, help='number of iterations')  # 迭代次数
parser.add_argument('--popsize', type=int, default=10, help='population size')  # 人口数量
args = parser.parse_args()
d = args.d
iters = args.iters
popsize = args.popsize
GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
# 忽略了警告错误的输出
warnings.simplefilter("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = {0: 'Non_Epileptic', 1: 'Epileptic'}


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def perturb(x):
    Zi = ZZ.copy()  # 5,22,4
    ddd = 0.5
    YY = Zi*0
    zzi = Zi*0
    pixs = np.array(np.split(x, len(x) / 6)).astype(int)  # pixs.shape(3, 6)
    loc = (pixs[:, 0], pixs[:, 1])
    val = pixs[:, 2:6]/10
    YY_val = sign(val)
    for i in range(val.shape[0]):
        for e in range(val.shape[1]):
            if abs(val[i][e]) > ddd:
                YY_val[i][e] = val[i][e] - sign(val[i][e]) * ddd
                val[i][e] = sign(val[i][e]) * ddd
    zzi[loc] = val
    YY[loc] = YY_val
    YY = YY.swapaxes(1, 2)
    zzi = zzi.swapaxes(1, 2)
    ZZi = Zi.swapaxes(1, 2)
    for i in range(YY.shape[0]):
        for e in range(YY.shape[1]):
                YY[i][e] = sign(ZZi[i][e]) * abs(YY[i][e]).sum()/22
    # 取决于Zi怎么插值回图
    Zi = create_inter(zzi) + create_inter(YY)
    Zi = np.clip(Zi, -ddd, ddd)
    Zd = X[0].cpu().detach().numpy() + Zi
    adv_img = torch.tensor(Zd, dtype=torch.float32)
    return adv_img, Zi


def optimize(x):
    adv_img, adv = perturb(x)
    inp = adv_img.float().unsqueeze(0)
    out = test_net(inp.cuda())  # [[-0.0634, -3.0034, -6.5402, -4.5706]], grad_fn=<LogSoftmaxBackward>)
    prob = softmax(out.cpu().data.numpy()[0]) # [0.938584   0.04962    0.00144424 0.01035171]
    return prob[pred_orig]  # 0.938584


def callback(x, convergence):
    global pred_adv, prob_adv
    adv_img, adv = perturb(x)
    inp = adv_img.float().unsqueeze(0)
    out = test_net(inp.cuda())
    prob = softmax(out.cpu().data.numpy()[0])
    pred_adv = np.argmax(prob)
    prob_adv = prob[pred_adv]
    if pred_adv != pred_orig:
        print('Attack successful..')
        print('Prob [%s]: %f' % (class_names[pred_adv], prob_adv))
        print()
        return True
    else:
        print('Prob [%s]: %f' % (class_names[pred_orig], prob[pred_orig]))


def create_inter(data):
    ch = ['AF7', 'FT7', 'T7', 'PO7', 'AF3', 'FC3', 'CP3', 'PO3', 'AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8', 'T8', 'PO8',
          'FCz', 'CPz', 'P7', 'T9', 'T10', 'P8']
    fake_info = mne.create_info(ch_names=ch, sfreq=256., ch_types='eeg')
    montage = make_standard_montage('standard_1020')
    ZZi = np.zeros([5, 4, 22, 22])
    for i in range(data.shape[0]):
        for c in range(data.shape[1]):
            da = data[i][c].reshape(-1, 1)
            fake_evoked = mne.EvokedArray(da, fake_info)
            fake_evoked.set_montage(montage)
            Zi = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, show=False)
            Zi = np.nan_to_num(Zi)
            ZZi[i][c] = Zi
    return ZZi


test_Images = sio.loadmat("topomap_test.mat")["data"]
test_Label = sio.loadmat("test_label.mat")["label"][0]
test_Label = torch.tensor(test_Label, dtype=torch.long)
Test = EEGImagesDataset(test_Label, test_Images)

x_test = sio.loadmat("FFT_test_data.mat")["data"]
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
PATH = r'D:\SeizuresCounter\MaxCNN_model.pth'
test_net = MaxCNN().cuda()
test_net.load_state_dict(torch.load(PATH))
test_net.eval()

loss = nn.CrossEntropyLoss()
pred_adv = 0
prob_adv = 0

if __name__ == '__main__':
    correct = 0
    eps = 0.001
    i = 0
    SUM_1 = 0
    for data, target in Test:
        Y = torch.tensor(target, dtype=torch.float32)
        X = torch.from_numpy(data).float().unsqueeze(0)
        X, Y = X.to(device), Y.to(device)
        X, Y = Variable(X, requires_grad=True), Variable(Y)
        prob_orig = softmax(test_net(X).cpu().data.numpy()[0])
        pred_orig = np.argmax(prob_orig)
        labels = Y.long()
        outputs = test_net(X)
        test_net.zero_grad()
        labels = torch.unsqueeze(labels, 0)
        cost = loss(outputs, labels)
        cost.backward()
        epss = eps * X.grad.sign()
        test = x_test[i]
        ZZ = np.zeros([5, 4, 22])
        for d in range(X.shape[1]):
            for c in range(X.shape[2]):
                data = epss[0][d][c].flatten()
                f = interpolate.CloughTocher2DInterpolator(list(zip(XY[:, 0], XY[:, 1])), data.cpu())
                znew = f(pos[:, 0], pos[:, 1])
                ZZ[d][c] = znew
        ZZ = ZZ.swapaxes(1, 2)  # 5,22,4
        bounds = [(0, 5), (0, 22), (-110, 110), (-110, 110), (-110, 110), (-110, 110)] * 5  # 22， 66 ，110
        result = differential_evolution(optimize, bounds, maxiter=iters, popsize=popsize, mutation=(0.5, 1.5), tol=1e-5, callback=callback)
        adv_img, adv = perturb(result.x)
        ZZ_adv = np.zeros([5, 4, 22])
        for d in range(X.shape[1]):
            for c in range(X.shape[2]):
                data = adv[d][c].flatten()
                f = interpolate.CloughTocher2DInterpolator(list(zip(XY[:, 0], XY[:, 1])), data)
                znew = f(pos[:, 0], pos[:, 1])
                ZZ_adv[d][c] = znew

        test = test + ZZ_adv
        Zd = create_inter(test)
        Zd = torch.tensor(Zd, dtype=torch.float32)
        attack_images = Zd.to(torch.float32).unsqueeze(0)
        attack_images = attack_images.to(device)

        outputs = test_net(attack_images)
        prob = softmax(outputs.cpu().detach().numpy()[0])
        SUM = 0
        o = 0
        for v in range(X.shape[0]):
            for t in range(X.shape[1]):
                for b in range(X.shape[2]):
                    RMSE = mean_squared_error(X[v][t][b].cpu() .detach().numpy().flatten(),
                                              attack_images[v][t][b].cpu() .detach().numpy().flatten()) ** 0.5
                    SUM = SUM + RMSE
                    o = o + 1
        SUM_1 = SUM_1 + SUM / o
        print('Prob [%s]: %f --> Prob[%s]: %f' % (
            class_names[pred_orig], prob_orig[pred_orig], class_names[pred_adv], prob_adv))
        correct += (pred_orig != pred_adv).sum().item()  # 攻击成功的数目
        i = i + 1
        print(labels, pred_orig, pred_adv, correct, i, (correct/i))
        print(SUM / o, SUM_1/i)
    print('Accuracy of the network on the  test images: %d %%' % (
            100 * correct / len(test_Label)))
    print(SUM_1/i)