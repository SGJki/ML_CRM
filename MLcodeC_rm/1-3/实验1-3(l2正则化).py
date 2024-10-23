import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.nn.functional import cross_entropy
from torchvision import transforms

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


# 绘制图像的代码
def picture(name, trainl, testl, type='Loss'):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.title(name)  # 命名
    plt.plot(trainl, c='g', label='Train ' + type)
    plt.plot(testl, c='r', label='Test ' + type)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


class MyNet3():
    def __init__(self):
        # 设置隐藏层和输出层的节点数
        num_inputs, num_hiddens, num_outputs = 28 * 28, 256, 10  # 十分类问题
        w_1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float32,
                           requires_grad=True)
        b_1 = torch.zeros(num_hiddens, dtype=torch.float32, requires_grad=True)
        w_2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float32,
                           requires_grad=True)
        b_2 = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)
        self.params = [w_1, b_1, w_2, b_2]

        # 定义模型结构
        self.input_layer = lambda x: x.view(x.shape[0], -1)
        self.hidden_layer = lambda x: self.my_relu(torch.matmul(x, w_1.t()) + b_1)
        self.output_layer = lambda x: torch.matmul(x, w_2.t()) + b_2

    def my_relu(self, x):
        return torch.max(input=x, other=torch.tensor(0.0))

    # 定义前向传播
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


def mySGD(params, lr, batchsize):
    for param in params:
        param.data -= lr * param.grad / batchsize

# 定义L2正则化函数
def l2_penalty(w1, w2):
    return ((w1 ** 2).sum() + (w2 ** 2).sum()) / 2


# 训练
model3 = MyNet3()  # logistics模型
criterion = cross_entropy  # 损失函数
lr = 0.15  # 学习率
epochs = 100  # 训练轮数
train_all_loss3 = []  # 记录训练集上得loss变化
test_all_loss3 = []  # 记录测试集上的loss变化
train_ACC13, test_ACC13 = [], []  # 记录正确的个数
begintime3 = time.time()

# reg_lambd = 5
reg_lambd = 0
# reg_lambd = 2
for epoch in range(epochs):
    train_l, train_acc_num = 0, 0
    for data, labels in traindataloader3:
        pred = model3.forward(data)
        train_each_loss = criterion(pred, labels) + reg_lambd * l2_penalty(model3.params[0],  # L2正则化
                                                                           model3.params[2])  # 计算每次的损失值
        train_l += train_each_loss.item()
        train_each_loss.backward()  # 反向传播
        mySGD(model3.params, lr, 128)  # 使用小批量随机梯度下降迭代模型参数
        # 梯度清零
        train_acc_num += (pred.argmax(dim=1) == labels).sum().item()
        for param in model3.params:
            param.grad.data.zero_()
        # print(train_each_loss)
    train_all_loss3.append(train_l)  # 添加损失值到列表中
    train_ACC13.append(train_acc_num / len(traindataset3))  # 添加准确率到列表中
    with torch.no_grad():
        test_l, test_acc_num = 0, 0
        for data, labels in testdataloader3:
            pred = model3.forward(data)
            test_each_loss = criterion(pred, labels)
            test_l += test_each_loss.item()
            test_acc_num += (pred.argmax(dim=1) == labels).sum().item()
        test_all_loss3.append(test_l)
        test_ACC13.append(test_acc_num / len(testdataset3))  # # 添加准确率到列表中
    if epoch == 0 or (epoch + 1) % 4 == 0:
        print('epoch: %d | train loss:%.5f | test loss:%.5f | train acc: %.2f | test acc: %.2f'
              % (epoch + 1, train_l, test_l, train_ACC13[-1], test_ACC13[-1]))
endtime3 = time.time()
print("手动实现前馈网络-多分类实验 %d轮 总用时: %.3f" % (epochs, endtime3 - begintime3))
picture('前馈网络-多分类-loss', train_all_loss3, test_all_loss3)
plt.show()
picture('前馈网络-多分类—ACC', train_ACC13, test_ACC13, type='ACC')
plt.show()
