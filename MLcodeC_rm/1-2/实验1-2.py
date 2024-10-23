import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy


# 构二分类数据集合
data_num, train_num, test_num = 10000, 7000, 3000  # 分别为样本总数量，训练集样本数量和测试集样本数量
# 第一个数据集 符合均值为 0.5 标准差为1 得分布
features1 = torch.normal(mean=0.2, std=2, size=(data_num, 200), dtype=torch.float32)
labels1 = torch.ones(data_num)
# 第二个数据集 符合均值为 -0.5 标准差为1的分布
features2 = torch.normal(mean=-0.2, std=2, size=(data_num, 200), dtype=torch.float32)
labels2 = torch.zeros(data_num)

# 构建训练数据集
train_features2 = torch.cat((features1[:train_num], features2[:train_num]), dim=0)  # size torch.Size([14000, 200])
train_labels2 = torch.cat((labels1[:train_num], labels2[:train_num]), dim=-1)  # size  torch.Size([6000, 200])
# 构建测试数据集
test_features2 = torch.cat((features1[train_num:], features2[train_num:]), dim=0)  # torch.Size([14000])
test_labels2 = torch.cat((labels1[train_num:], labels2[train_num:]), dim=-1)  # torch.Size([6000])
batch_size = 128
# Build the training and testing dataset
traindataset2 = torch.utils.data.TensorDataset(train_features2, train_labels2)
testdataset2 = torch.utils.data.TensorDataset(test_features2, test_labels2)
traindataloader2 = torch.utils.data.DataLoader(dataset=traindataset2, batch_size=batch_size, shuffle=True)
testdataloader2 = torch.utils.data.DataLoader(dataset=testdataset2, batch_size=batch_size, shuffle=True)


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


print(
    f'二分类数据集 样本总数量{len(traindataset2) + len(testdataset2)},训练样本数量{len(traindataset2)},测试样本数量{len(testdataset2)}')


# 定义自己的前馈神经网络（二分类任务）
class MyNet2():
    def __init__(self):
        # 设置隐藏层和输出层的节点数
        num_inputs, num_hiddens, num_outputs = 200, 256, 1
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
        self.fn_logistic = self.logistic

    def my_relu(self, x):
        return torch.max(input=x, other=torch.tensor(0.0))

    def logistic(self, x):  # 定义logistic函数
        x = 1.0 / (1.0 + torch.exp(-x))
        return x

    # 定义前向传播
    def forward(self, x):
        x = self.input_layer(x)
        x = self.my_relu(self.hidden_layer(x))
        x = self.fn_logistic(self.output_layer(x))
        return x


def mySGD(params, lr):
    for param in params:
        param.data -= lr * param.grad


# 训练
model2 = MyNet2()
lr = 0.01  # 学习率
epochs = 40  # 训练轮数
train_all_loss2 = []  # 记录训练集上得loss变化
test_all_loss2 = []  # 记录测试集上的loss变化
train_Acc12, test_Acc12 = [], []
begintime2 = time.time()
for epoch in range(epochs):
    train_l, train_epoch_count = 0, 0
    for data, labels in traindataloader2:
        pred = model2.forward(data)
        train_each_loss = binary_cross_entropy(pred.view(-1), labels.view(-1))  # 计算每次的损失值
        train_l += train_each_loss.item()
        train_each_loss.backward()  # 反向传播
        mySGD(model2.params, lr)  # 使用随机梯度下降迭代模型参数
        # 梯度清零
        for param in model2.params:
            param.grad.data.zero_()
        # print(train_each_loss)
        train_epoch_count += (torch.tensor(np.where(pred > 0.5, 1, 0)).view(-1) == labels).sum()
    train_Acc12.append((train_epoch_count / len(traindataset2)).item())
    train_all_loss2.append(train_l)  # 添加损失值到列表中
    with torch.no_grad():
        test_l, test_epoch_count = 0, 0
        for data, labels in testdataloader2:
            pred = model2.forward(data)
            test_each_loss = binary_cross_entropy(pred.view(-1), labels.view(-1))
            test_l += test_each_loss.item()
            test_epoch_count += (torch.tensor(np.where(pred > 0.5, 1, 0)).view(-1) == labels.view(-1)).sum()
        test_Acc12.append((test_epoch_count / len(testdataset2)).item())
        test_all_loss2.append(test_l)
    if epoch == 0 or (epoch + 1) % 4 == 0:
        print('epoch: %d | train loss:%.5f | test loss:%.5f | train acc:%.5f | test acc:%.5f' % (
            epoch + 1, train_all_loss2[-1], test_all_loss2[-1], train_Acc12[-1], test_Acc12[-1]))
endtime2 = time.time()
print("手动实现前馈网络-二分类实验 %d轮 总用时: %.3f" % (epochs, endtime2 - begintime2))
picture('前馈网络-二分类-loss', train_all_loss2, test_all_loss2)
plt.show()
picture('前馈网络-二分类-ACC', train_Acc12, test_Acc12, type='ACC')
plt.show()
