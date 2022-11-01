from torch.utils.data import DataLoader
from Utils import *
from Models import *
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(0)
# 忽略了警告错误的输出
warnings.simplefilter("ignore")


Images = sio.loadmat("topomap_train.mat")["data"]
Label = sio.loadmat("train_label.mat")["label"][0]
test_Images = sio.loadmat("topomap_test.mat")["data"]
test_Label = sio.loadmat("test_label.mat")["label"][0]
Label = torch.tensor(Label, dtype=torch.long)
test_Label = torch.tensor(test_Label, dtype=torch.long)

Train = EEGImagesDataset(Label, Images)
Test = EEGImagesDataset(test_Label, test_Images)
train_loader = torch.utils.data.DataLoader(Train, batch_size=64,
                                           shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(Test, batch_size=64,
                                          shuffle=True, num_workers=0)


"""
   MaxCNN_best_model.pth：93%
   TempCNN_best_model.pth：93% 
   LSTM_best_model.pth：94%  
   Mix_best_model.pth：94% 
"""
"""训练模型"""
net = Mix()  # 92%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
# best_loss统计，初始化为正无穷
best_loss = float('inf')
total_step = len(train_loader)
train_accs = []
train_loss = []
test_accs = []
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
if __name__ == '__main__':
    print("\n\n\n\n Mix \n\n\n\n")
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
    torch.save(net.state_dict(), 'Mix_model.pth')
    print('Finished Training')
    train_iters = range(len(train_accs))
    draw_train_process('training', train_iters, train_loss, train_accs, 'training loss', 'training acc')

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(64)))
    net.eval()
    outputs = net(images.to(torch.float32).cuda())
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % predicted[j]for j in range(64)))

    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images.to(torch.float32).cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # labels 的长度
            correct += (predicted == labels).sum().item()  # 预测正确的数目
        print('Accuracy of the network on the  test images: %d %%' % (
                100 * correct / total))


