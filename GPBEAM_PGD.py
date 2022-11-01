from torch.utils.data import DataLoader
from Utils import *
from Models import *
import os
from scipy import interpolate
import mne
from mne.channels import make_standard_montage
import pandas as pd
import warnings
import argparse
from mne.viz.topomap import MINE_plot_topomap
from sklearn.metrics import mean_squared_error
torch.manual_seed(1234)
np.random.seed(1234)
parser = argparse.ArgumentParser(description='PyTorch MNIST PGD Attack Evaluation')
parser.add_argument('--gpuid', default=0,
                    help='gpuid')
parser.add_argument('--epsilon', default=0.1, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.01, type=float,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--momentum-PGD', default=0, type=int,
                    help='whether perform momentum PGD')

args = parser.parse_args()
is_momentum = (args.momentum_PGD == 1)
GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.simplefilter("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  #


def create_montage(data):
    ch = ['AF7', 'FT7', 'T7', 'PO7', 'AF3', 'FC3', 'CP3', 'PO3', 'AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8', 'T8', 'PO8',
          'FCz', 'CPz', 'P7', 'T9', 'T10', 'P8']
    fake_info = mne.create_info(ch_names=ch, sfreq=256., ch_types='eeg')
    montage = make_standard_montage('standard_1020')
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 4), gridspec_kw=dict(top=0.9), sharex=True, sharey=True)
    for g in range(data.shape[0]):
        da = data[g].reshape(-1, 1)
        fake_evoked = mne.EvokedArray(da, fake_info)
        fake_evoked.set_montage(montage)
        if g >= 4:
            MINE_plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, vmin=0, vmax=50, cmap='RdBu', axes=ax[1][g-4], show=False)
        else:
            MINE_plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, vmin=0, vmax=50, cmap='RdBu',  axes=ax[0][g], show=False)
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


def _pgd_whitebox(model, X, y, test_data,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    y = y.long()
    X_pgd = X
    X_pgd = X_pgd.cuda()
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-step_size, step_size)
        X_pgd = X_pgd.cuda() + random_noise.to(device)
    for _ in range(num_steps):
        X_pgd.requires_grad = True
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y.cuda())
        loss.backward()
        eta = step_size * X_pgd.grad.sign()
        X_pgd = X_pgd.data + eta
        eta = torch.clamp(X_pgd.cuda() - X.cuda(), -epsilon, epsilon)

        ZZ = np.zeros([eta.shape[0], 5, 4, 22])
        for v in range(eta.shape[0]):
            for d in range(eta.shape[1]):
                for c in range(eta.shape[2]):
                    data = eta[v][d][c].flatten()
                    f = interpolate.CloughTocher2DInterpolator(list(zip(XY[:, 0], XY[:, 1])), data.cpu())
                    znew = f(pos[:, 0], pos[:, 1])
                    ZZ[v][d][c] = znew
        ZZ_eta = np.zeros([eta.shape[0], 5, 4, 22, 22])
        for u in range(test_data.shape[0]):
            test_data[u] = test_data[u] + ZZ[u]
            # dd = np.concatenate([x_test[u][0], X_test[u][0]], axis=0)
            # create_montage(dd)
            ZZ_eta[u] = create_inter(test_data[u])
        ZZ_eta = torch.tensor(ZZ_eta, dtype=torch.float32)
        X_pgd = ZZ_eta.to(device)
    return X_pgd, test_data


def eval_adv_test_whitebox(model, device, test_loader, test_data):
    """
    evaluate model by white-box attack
    """
    global attack_images
    model.eval()
    i = 0
    correct = 0
    X_correct = 0
    total = 0
    dd =[]
    SUM_1 = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        y = torch.tensor(target, dtype=torch.float32)
        X = torch.tensor(data, dtype=torch.float32)
        X_outputs = model(X.cuda())
        _, X_predicted = torch.max(X_outputs, dim=1)
        attack_images, X_test = _pgd_whitebox(model, X, y, test_data[0 + i * 64:64 + i * 64])
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
    # sio.savemat('attack_PGD_data.mat', {'data': dd})


test_Images = sio.loadmat("topomap_test.mat")["data"]
test_Label = sio.loadmat("test_label.mat")["label"][0]
test_Label = torch.tensor(test_Label, dtype=torch.long)
Test = EEGImagesDataset(test_Label, test_Images)
test_loader = torch.utils.data.DataLoader(Test, batch_size=64,
                                          shuffle=False, num_workers=0)

test_data = sio.loadmat("FFT_test_data.mat")["data"]
XY = pd.read_csv("XY.csv")
XY = np.array(XY)
pos = pd.read_csv("pos.csv")
pos = np.array(pos)


if __name__ == '__main__':
    PATH = r'D:\SeizuresCounter\Mix_model.pth'
    test_net = Mix().cuda()
    test_net.load_state_dict(torch.load(PATH))
    eval_adv_test_whitebox(test_net, device, test_loader, test_data)
