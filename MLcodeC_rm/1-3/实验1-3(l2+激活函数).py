import time
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.nn as nn
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果有gpu则在gpu上计算 加快计算速度
# 定义多分类数据集 - train_dataloader - test_dataloader
batch_size = 128
# Build the training and testing dataset
# 下载MNIST手写数据集
traindataset3 = torchvision.datasets.MNIST(root='./Datasets/MNIST/Train', train=True, download=True,
                                           transform=transforms.ToTensor())
testdataset3 = torchvision.datasets.MNIST(root='./Datasets/MNIST/Test', train=False, download=True,
                                          transform=transforms.ToTensor())
traindataloader3 = torch.utils.data.DataLoader(traindataset3, batch_size=batch_size, shuffle=True)
testdataloader3 = torch.utils.data.DataLoader(testdataset3, batch_size=batch_size, shuffle=False)

print(
    f'多分类数据集 样本总数量{len(traindataset3) + len(testdataset3)},训练样本数量{len(traindataset3)},测试样本数量{len(testdataset3)}')


# 定义自己的前馈神经网络
class MyNet3(nn.Module):
    """
    参数：  num_input：输入每层神经元个数，为一个列表数据
            num_hiddens：隐藏层神经元个数
            num_outs： 输出层神经元个数
            num_hiddenlayer : 隐藏层的个数
    """

    def __init__(self, num_hiddenlayer=1, num_inputs=28 * 28, num_hiddens=[256], num_outs=10, act='relu'):
        super(MyNet3, self).__init__()
        # 设置隐藏层和输出层的节点数
        self.num_inputs, self.num_hiddens, self.num_outputs = num_inputs, num_hiddens, num_outs  # 十分类问题

        # 定义模型结构
        self.input_layer = nn.Flatten()
        # 若只有一层隐藏层
        if num_hiddenlayer == 1:
            self.hidden_layers = nn.Linear(self.num_inputs, self.num_hiddens[-1])
        else:  # 若有多个隐藏层
            self.hidden_layers = nn.Sequential()
            self.hidden_layers.add_module("hidden_layer1", nn.Linear(self.num_inputs, self.num_hiddens[0]))
            for i in range(0, num_hiddenlayer - 1):
                name = str('hidden_layer' + str(i + 2))
                self.hidden_layers.add_module(name, nn.Linear(self.num_hiddens[i], self.num_hiddens[i + 1]))
        self.output_layer = nn.Linear(self.num_hiddens[-1], self.num_outputs)
        # 指代需要使用什么样子的激活函数
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'elu':
            self.act = nn.ELU()
        print(f'你本次使用的激活函数为 {act}')

    def logistic(self, x):  # 定义logistic函数
        x = 1.0 / (1.0 + torch.exp(-x))
        return x

    # 定义前向传播
    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(self.hidden_layers(x))
        x = self.output_layer(x)
        return x


# 训练
# 使用默认的参数即： num_inputs=28*28,num_hiddens=256,num_outs=10,act='relu'
model3 = MyNet3()
model3 = model3.to(device)


# 训练过程函数
def train_and_test(model=model3):
    MyModel = model
    print(MyModel)
    optimizer = SGD(MyModel.parameters(), lr=0.01)  # 优化函数
    epochs = 40  # 训练轮数
    criterion = CrossEntropyLoss()  # 损失函数
    train_all_loss23 = []  # 记录训练集上得loss变化
    test_all_loss23 = []  # 记录测试集上的loss变化
    train_ACC23, test_ACC23 = [], []
    begintime23 = time.time()
    for epoch in range(epochs):
        train_l, train_epoch_count, test_epoch_count = 0, 0, 0
        for data, labels in traindataloader3:
            data, labels = data.to(device), labels.to(device)
            pred = MyModel(data)
            train_each_loss = criterion(pred, labels.view(-1))  # 计算每次的损失值
            optimizer.zero_grad()  # 梯度清零
            train_each_loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            train_l += train_each_loss.item()
            train_epoch_count += (pred.argmax(dim=1) == labels).sum()
        train_ACC23.append(train_epoch_count.cpu() / len(traindataset3))
        train_all_loss23.append(train_l)  # 添加损失值到列表中
        with torch.no_grad():
            test_loss, test_epoch_count = 0, 0
            for data, labels in testdataloader3:
                data, labels = data.to(device), labels.to(device)
                pred = MyModel(data)
                test_each_loss = criterion(pred, labels)
                test_loss += test_each_loss.item()
                test_epoch_count += (pred.argmax(dim=1) == labels).sum()
            test_all_loss23.append(test_loss)
            test_ACC23.append(test_epoch_count.cpu() / len(testdataset3))
        if epoch == 0 or (epoch + 1) % 4 == 0:
            print('epoch: %d | train loss:%.5f | test loss:%.5f | train acc:%5f test acc:%.5f:' % (
                epoch + 1, train_all_loss23[-1], test_all_loss23[-1],
                train_ACC23[-1], test_ACC23[-1]))
    endtime23 = time.time()
    print("torch.nn实现前馈网络-多分类任务 %d轮 总用时: %.3fs" % (epochs, endtime23 - begintime23))
    # 返回训练集和测试集上的 损失值 与 准确率
    return train_all_loss23, test_all_loss23, train_ACC23, test_ACC23


train_all_loss23, test_all_loss23, train_ACC23, test_ACC23 = train_and_test(model=model3)


def ComPlot(datalist, title='1', ylabel='Loss', flag='act'):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(datalist[0], label='Tanh' if flag == 'act' else '[128]')
    plt.plot(datalist[1], label='Sigmoid' if flag == 'act' else '[512 256]')
    plt.plot(datalist[2], label='ELu' if flag == 'act' else '[512 256 128 64]')
    plt.plot(datalist[3], label='Relu' if flag == 'act' else '[256]')
    plt.legend()
    plt.grid(True)


# 使用多分类的模型定义其激活函数为 Tanh
model31 = MyNet3(1, 28 * 28, [256], 10, act='tanh')
model31 = model31.to(device)  # 若有gpu则放在gpu上训练
# 调用定义的训练函数，避免重复编写代码
train_all_loss31, test_all_loss31, train_ACC31, test_ACC31 = train_and_test(model=model31)
# 使用多分类的模型定义其激活函数为 Sigmoid
model32 = MyNet3(1, 28 * 28, [256], 10, act='sigmoid')
model32 = model32.to(device)  # 若有gpu则放在gpu上训练
# 调用定义的训练函数，避免重复编写代码
train_all_loss32, test_all_loss32, train_ACC32, test_ACC32 = train_and_test(model=model32)
# 使用多分类的模型定义其激活函数为 ELU
model33 = MyNet3(1, 28 * 28, [256], 10, act='elu')
model33 = model33.to(device)  # 若有gpu则放在gpu上训练
# 调用定义的训练函数，避免重复编写代码
train_all_loss33, test_all_loss33, train_ACC33, test_ACC33 = train_and_test(model=model33)

plt.figure(figsize=(16, 3))
plt.subplot(141)
ComPlot([train_all_loss31, train_all_loss32, train_all_loss33, train_all_loss23], title='Train_Loss')
plt.subplot(142)
ComPlot([test_all_loss31, test_all_loss32, test_all_loss33, test_all_loss23], title='Test_Loss')
plt.subplot(143)
ComPlot([train_ACC31, train_ACC32, train_ACC33, train_ACC23], title='Train_ACC')
plt.subplot(144)
ComPlot([test_ACC31, test_ACC32, test_ACC33, test_ACC23], title='Test_ACC')
plt.show()
