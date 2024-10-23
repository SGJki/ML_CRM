import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss


# 构建回归数据集合 - train_dataloader1, test_dataloader1
data_num, train_num, test_num = 10000, 7000, 3000  # 分别为样本总数量，训练集样本数量和测试集样本数量
true_w, true_b = 0.0056 * torch.ones(500, 1), 0.028
features = torch.randn(10000, 500)
labels = torch.matmul(features, true_w) + true_b  # 按高斯分布
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# 划分训练集和测试集
train_features, test_features = features[:train_num, :], features[train_num:, :]
train_labels, test_labels = labels[:train_num], labels[train_num:]
batch_size = 128
train_dataset1 = torch.utils.data.TensorDataset(train_features, train_labels)
test_dataset1 = torch.utils.data.TensorDataset(test_features, test_labels)
train_dataloader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=batch_size, shuffle=True)
test_dataloader1 = torch.utils.data.DataLoader(dataset=test_dataset1, batch_size=batch_size, shuffle=True)


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
    f'回归数据集   样本总数量{len(train_dataset1) + len(test_dataset1)},训练样本数量{len(train_dataset1)},测试样本数量{len(test_dataset1)}')


# 定义自己的前馈神经网络（回归任务）
class MyNet1():
    def __init__(self):
        # 设置隐藏层和输出层的节点数
        num_inputs, num_hiddens, num_outputs = 500, 256, 1
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

    def forward(self, x):
        x = self.input_layer(x)
        x = self.my_relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


def mySGD(params, lr, batchsize):
    for param in params:
        param.data -= lr * param.grad / batchsize


def mse(pred, true):
    ans = torch.sum((true - pred) ** 2) / len(pred)
    # print(ans)
    return ans


# 训练
model1 = MyNet1()  # logistics模型
criterion = CrossEntropyLoss()  # 损失函数
lr = 0.05  # 学习率
batchsize = 128
epochs = 40  # 训练轮数
train_all_loss1 = []  # 记录训练集上得loss变化
test_all_loss1 = []  # 记录测试集上的loss变化
begintime1 = time.time()
for epoch in range(epochs):
    train_l = 0
    for data, labels in train_dataloader1:
        pred = model1.forward(data)
        train_each_loss = mse(pred.view(-1, 1), labels.view(-1, 1))  # 计算每次的损失值
        train_each_loss.backward()  # 反向传播
        mySGD(model1.params, lr, batchsize)  # 使用小批量随机梯度下降迭代模型参数
        # 梯度清零
        train_l += train_each_loss.item()
        for param in model1.params:
            param.grad.data.zero_()
        # print(train_each_loss)
    train_all_loss1.append(train_l)  # 添加损失值到列表中
    with torch.no_grad():
        test_loss = 0
        for data, labels in train_dataloader1:
            pred = model1.forward(data)
            test_each_loss = mse(pred, labels)
            test_loss += test_each_loss.item()
        test_all_loss1.append(test_loss)
    if epoch == 0 or (epoch + 1) % 4 == 0:
        print('epoch: %d | train loss:%.5f | test loss:%.5f' % (epoch + 1, train_all_loss1[-1], test_all_loss1[-1]))
endtime1 = time.time()
print("手动实现前馈网络-回归实验 %d轮 总用时: %.3fs" % (epochs, endtime1 - begintime1))
picture('前馈网络-回归-Loss', train_all_loss1, test_all_loss1)
plt.show()