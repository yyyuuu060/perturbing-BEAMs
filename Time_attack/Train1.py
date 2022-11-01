from torch.utils.data import DataLoader, random_split
from Utils import *
from Models import *
import warnings
import os
import random
from sklearn.metrics import mean_squared_error

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(0)

# 忽略了警告错误的输出
warnings.simplefilter("ignore")

Images = sio.loadmat("train_data.mat")["data"]  # (5969, 22, 1281)
Label = sio.loadmat("train_label.mat")["label"][0]
test_Images = sio.loadmat("test_data.mat")["data"]
test_Label = sio.loadmat("test_label.mat")["label"][0]
attack_time_data = sio.loadmat("attack_time_MaxCNN_1.5.mat")["data"]

Label = torch.tensor(Label, dtype=torch.long)
test_Label = torch.tensor(test_Label, dtype=torch.long)

# 训练集转换
nn_train = np.zeros([Images.shape[0], 5, 22, 256])
for i in range(Images.shape[0]):
    for r in range(Images.shape[1]):
        for e in range(5):
            nn_train[i][e][r] = Images[i][r][0 + 256 * e:256 + 256 * e]
nn_train = nn_train[:, :, np.newaxis, :, :]  #

# 测试集转换
test_nn = np.zeros([test_Images.shape[0], 5, 22, 256])
for i in range(test_Images.shape[0]):
    for r in range(test_Images.shape[1]):
        for e in range(5):
            test_nn[i][e][r] = test_Images[i][r][0 + 256 * e:256 + 256 * e]
test_nn = test_nn[:, :, np.newaxis, :, :]  #

# BEAM转换回来的EEG对抗样本
attack_nn = np.zeros([attack_time_data.shape[0], 5, 22, 256])
for i in range(attack_time_data.shape[0]):
    for r in range(attack_time_data.shape[1]):
        for e in range(5):
            attack_nn[i][e][r] = attack_time_data[i][r][0 + 256 * e:256 + 256 * e]
attack_nn = attack_nn[:, :, np.newaxis, :, :]  #

o = 0
SUM = 0
for i in range(attack_nn.shape[0]):
    for n in range(attack_nn.shape[1]):
            RMSE = mean_squared_error(attack_nn[i][n].flatten(), test_nn[i][n].flatten()) ** 0.5
            SUM = SUM + RMSE
            o = o + 1
print(SUM/o)
Train = EEGImagesDataset(Label, nn_train)
Test = EEGImagesDataset(test_Label, test_nn)
attack_data = EEGImagesDataset(test_Label, attack_nn)

attack_loader = torch.utils.data.DataLoader(attack_data, batch_size=16,
                                            shuffle=False, num_workers=0)
train_loader = torch.utils.data.DataLoader(Train, batch_size=16,
                                           shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(Test, batch_size=16,
                                          shuffle=False, num_workers=0)


"""训练模型"""
net = MaxCNN()
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
# best_loss统计，初始化为正无穷
best_loss = float('inf')
total_step = len(train_loader)
train_accs = []
train_loss = []
test_accs = []
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

if __name__ == '__main__':
    print("\n\n\n\n MaxCNN \n\n\n\n")
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs.to(torch.float32).cuda())
            loss = criterion(outputs, labels)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss.append(loss.item())
            # 训练曲线的绘制 一个batch中的准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)  # labels 的长度
            correct = (predicted == labels).sum().item()  # 预测正确的数目
            train_accs.append(100 * correct / total)
            if (i + 1) % 100 == 0:
                print('Epoch[{}/{}],Step[{},{}],Loss:{:.4f},Accuracy:{:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))
    torch.save(net.state_dict(), 'MaxCNN_model.pth')

    all = np.zeros([1297])
    all_a = np.zeros([1297])
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        correct = 0
        total = 0
        i = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            c = images.shape[0]
            outputs = net(images.to(torch.float32).cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # labels 的长度
            # correct = (predicted == labels).sum().item()
            correct += (predicted == labels).sum().item()  # 预测正确的数目
            all[0 + i * c:16 + i * c] = predicted.cpu()
            i = i + 1
        print('Accuracy of the network on the  test images: %d %%' % (
                100 * correct / total))

    # BEAM转换回来的EEG对抗样本评测
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        correct = 0
        total = 0
        i = 0
        for data in attack_loader:
            images, labels = data[0].to(device), data[1].to(device)
            c = images.shape[0]
            outputs = net(images.to(torch.float32).cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # labels 的长度
            # correct = (predicted == labels).sum().item()
            correct += (predicted == labels).sum().item()  # 预测正确的数目
            all_a[0 + i * c:16 + i * c] = predicted.cpu()
            i = i + 1
        print('Accuracy of the network on the  test images: %d %%' % (
                100 * correct / total))
        print('success of the network on the  test images: %d %%' % (
                100 * (all != all_a).sum().item() / total))
