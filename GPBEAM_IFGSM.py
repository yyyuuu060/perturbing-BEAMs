from torch.utils.data import DataLoader, random_split
from Utils import *
from Models import *
import os
from scipy import interpolate
from sklearn.metrics import mean_squared_error
import mne
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
# 忽略了警告错误的输出
warnings.simplefilter("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  #

# Paint the BEAMs and BEAMs adversarial sample
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
            MINE_plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, vmin=0, vmax=10, cmap='RdBu', axes=ax[1][g-4], show=False)
        else:
            MINE_plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, vmin=0, vmax=10, cmap='RdBu', axes=ax[0][g], show=False)
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

# The rhythmic power array adversarial samples are re-converted to BEAMs adversarial samples
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


def ifgsm_attack(model, loss, images, labels, test_data):
    for i in range(10):
        labels = labels.long()
        images = images.to(device)
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs, labels.cuda())
        cost.backward()
        eps = 0.01 * images.grad.sign()
        ZZ = np.zeros([eps.shape[0], 5, 4, 22])
        for v in range(eps.shape[0]):
            for d in range(eps.shape[1]):
                for c in range(eps.shape[2]):
                    data = eps[v][d][c].flatten()
                    f = interpolate.CloughTocher2DInterpolator(list(zip(XY[:, 0], XY[:, 1])), data.cpu())
                    znew = f(pos[:, 0], pos[:, 1])
                    ZZ[v][d][c] = znew
        ZZ_eps = np.zeros([eps.shape[0], 5, 4, 22, 22])
        for u in range(test_data.shape[0]):
            test_data[u] = test_data[u] + ZZ[u]
            # dd = np.concatenate([x_test[u][0], test_data[u][0]], axis=0) # Paint the BEAMs and BEAMs Adversarial sample
            # create_montage(dd)
            ZZ_eps[u] = create_inter(test_data[u])
        ZZ_eps = torch.tensor(ZZ_eps, dtype=torch.float32)
        images = ZZ_eps.to(device)
    return images, test_data


def eval_adv_test_iFGSM(model, device, test_loader, test_data):
    """
    evaluate model by white-box attack
    """
    model.eval()
    loss = nn.CrossEntropyLoss()
    print("Attack Image & Predicted Label")
    correct = 0
    X_correct = 0
    total = 0
    dd = []
    SUM_1 = 0
    for i, data in enumerate(test_loader, 0):
        data, target = data[0].to(device), data[1].to(device)
        y = torch.tensor(target, dtype=torch.float32)
        x = torch.tensor(data, dtype=torch.float32)
        X_outputs = model(x.cuda())
        _, X_predicted = torch.max(X_outputs, dim=1)
        attack_images, x_test = ifgsm_attack(model, loss, x, y, test_data[0 + i * 64:64 + i * 64])
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
        for v in range(x.shape[0]):
            for t in range(x.shape[1]):
                for b in range(x.shape[2]):
                    RMSE = mean_squared_error(x[v][t][b].cpu().detach().numpy().flatten(),  attack_images[v][t][b].cpu().detach().numpy().flatten()) ** 0.5
                    SUM = SUM + RMSE
                    o = o + 1
        SUM_1 = SUM_1 + SUM / o
        if i == 0:
            dd = x_test
        else:
            dd = np.concatenate([dd, x_test], axis=0)
    print(SUM_1 / (i+1))
    print('Accuracy of the network on the  test images: %d %%' % (
            100 * correct / total))
    print('Success rate of the network on the  test images: %d %%' % (
            100 * X_correct / total))
    # sio.savemat('attack_IFGSM_0.5.mat', {'data': dd})


test_Images = sio.loadmat("topomap_test.mat")["data"]
test_Label = sio.loadmat("test_label.mat")["label"][0]
test_Label = torch.tensor(test_Label, dtype=torch.long)
Test = EEGImagesDataset(test_Label, test_Images)
test_loader = torch.utils.data.DataLoader(Test, batch_size=64,
                                          shuffle=False, num_workers=0)

x_test = sio.loadmat("FFT_test_data.mat")["data"]
XY = np.array(pd.read_csv("XY.csv"))
pos = np.array(pd.read_csv("pos.csv"))

if __name__ == '__main__':
    PATH = r'C:\Users\HHC\Desktop\SeizuresCounter\Mix.pth'
    test_net = Mix().cuda()
    test_net.load_state_dict(torch.load(PATH))
    eval_adv_test_iFGSM(test_net, device, test_loader, x_test)


