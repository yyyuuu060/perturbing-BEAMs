import torch
from torch.utils.data import DataLoader, random_split
from Utils import *
from Models import *
import os
from scipy import interpolate
import mne
from sklearn.metrics import mean_squared_error
from mne.channels import make_standard_montage
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
from mne.viz.topomap import MINE_plot_topomap
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


def create_montage(data):
    ch = ['AF7', 'FT7', 'T7', 'PO7', 'AF3', 'FC3', 'CP3', 'PO3', 'AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8', 'T8', 'PO8',
          'FCz', 'CPz', 'P7', 'T9', 'T10', 'P8']
    fake_info = mne.create_info(ch_names=ch, sfreq=256., ch_types='eeg')
    montage = make_standard_montage('standard_1020')
    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(8, 4), gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)
    for g in range(data.shape[0]):
        da = data[g].reshape(-1, 1)
        fake_evoked = mne.EvokedArray(da, fake_info)
        fake_evoked.set_montage(montage)
        if g >= 4:
            MINE_plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, vmin=0, vmax=20, cmap='turbo',
                              axes=ax[1][g-4], show=False)
        else:
            MINE_plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, vmin=0, vmax=20, cmap='turbo',
                              axes=ax[0][g], show=False)
    ax[0][0].set_title('Delta', fontweight='bold')
    ax[0][1].set_title('Theta', fontweight='bold')
    ax[0][2].set_title('Alpha', fontweight='bold')
    ax[0][3].set_title('Beta', fontweight='bold')
    ax[1][0].set_title('Delta', fontweight='bold')
    ax[1][1].set_title('Theta', fontweight='bold')
    ax[1][2].set_title('Alpha', fontweight='bold')
    ax[1][3].set_title('Beta', fontweight='bold')
    plt.show()
    return 0


def create_inter(data):
    ch = ['AF7', 'FT7', 'T7', 'PO7', 'AF3', 'FC3', 'CP3', 'PO3', 'AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8', 'T8', 'PO8',
          'FCz', 'CPz', 'P7', 'T9','T10', 'P8']
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


def fgsm_attack(model, loss, images, labels, eps, x_test):
    labels = labels.long()
    images.requires_grad = True
    outputs = model(images.cuda())
    model.zero_grad()
    cost = loss(outputs, labels.cuda())
    cost.backward()
    eps = eps * images.grad.sign()
    noise = np.random.normal(0, 0.5 ** 0.5, eps.shape)
    eps = torch.from_numpy(noise)
    ZZ = np.zeros([eps.shape[0], 5, 4, 22])
    for v in range(eps.shape[0]):
        for d in range(eps.shape[1]):
            for c in range(eps.shape[2]):
                data = eps[v][d][c].flatten()
                f = interpolate.CloughTocher2DInterpolator(list(zip(XY[:, 0], XY[:, 1])), data.cpu())
                znew = f(pos[:, 0], pos[:, 1])
                ZZ[v][d][c] = znew
    ZZ_eps = np.zeros([eps.shape[0], 5, 4, 22, 22])
    for u in range(x_test.shape[0]):
        # dd = np.concatenate([x_test[u][0], x_test[u][0]+ZZ[u][0]], axis=0)
        # create_montage(dd)
        x_test[u] = x_test[u] + ZZ[u]
        ZZ_eps[u] = create_inter(x_test[u])
    sio.savemat('FFT_MaxCNN_0.5.mat', {'data': x_test})
    attack_images = torch.tensor(ZZ_eps, dtype=torch.float32)
    return attack_images, x_test


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
    eps = 0.5
    dd = []
    SUM_1 = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        y = torch.tensor(target, dtype=torch.float32)
        X = torch.tensor(data, dtype=torch.float32)
        X_outputs = model(X.cuda())
        _, X_predicted = torch.max(X_outputs, dim=1)
        attack_images, X_test = fgsm_attack(model, loss, X, y, eps, x_test[0 + i * 64:64 + i * 64])
        attack_images = attack_images.to(device)
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
            dd = X_test
        else:
            dd = np.concatenate([dd, X_test], axis=0)
        i = i + 1


    print(SUM_1/i)
    print('Accuracy of the network on the  test images: %d %%' % (
            100 * correct / total))
    print('Success rate of the network on the  test images: %d %%' % (
            100 * X_correct / total))
    sio.savemat('Mix_0.5.mat', {'data': dd})


test_Images = sio.loadmat("topomap_test.mat")["data"]
test_Label = sio.loadmat("test_label.mat")["label"][0]
test_Label = torch.tensor(test_Label, dtype=torch.long)
Test = EEGImagesDataset(test_Label, test_Images)
test_loader = torch.utils.data.DataLoader(Test, batch_size=64,
                                          shuffle=False, num_workers=0)

x_test = sio.loadmat("FFT_test_data.mat")["data"]
print(x_test.max(), x_test.min(),test_Images.max(),test_Images.min())
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
    PATH = r'D:\SeizuresCounter\Mix_model.pth'
    test_net = Mix().cuda()
    test_net.load_state_dict(torch.load(PATH))
    eval_adv_test_FGSM(test_net, device, test_loader, x_test)
    # PATH1 = r'D:\SeizuresCounter\MaxCNN_model.pth'
    # test_net1 = MaxCNN().cuda()
    # test_net1.load_state_dict(torch.load(PATH1))
    # eval_adv_test_FGSM(test_net, test_net1, device, test_loader, x_test)

# 0.689450272277809
# Accuracy of the network on the  test images: 84 %
# Success rate of the network on the  test images: 12 %

# 0.689450272277809
# Accuracy of the network on the  test images: 81 %
# Success rate of the network on the  test images: 16 %

# 0.689450272277809
# Accuracy of the network on the  test images: 84 %
# Success rate of the network on the  test images: 13 %

# 0.689450272277809
# Accuracy of the network on the  test images: 79 %
# Success rate of the network on the  test images: 17 %