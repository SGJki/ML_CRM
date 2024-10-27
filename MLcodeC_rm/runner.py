from utils.dataset import Dataset
from utils.train import train
from utils.draw import draw
from utils.net import Net, NetCls
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
from torch.optim import SGD
import random
import os
import torch
from absl import app


def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def regression():
    data_train, data_test = Dataset.data_reg()
    net_reg = Net()
    lr = 0.05
    optim = SGD(net_reg.params, lr)
    train_loss, test_loss = train(net_reg, data_train, data_test, loss=MSELoss(), lr=lr, optimizer=optim)
    draw('FNN for regression', train_loss, test_loss)


def logic_regression():
    data_train, data_test = Dataset.data_cls()
    net_log = NetCls()
    lr = 0.05
    optim = SGD(net_log.params, lr)
    train_loss, test_loss = train(net_log, data_train, data_test, loss=BCELoss(), lr=lr, optimizer=optim)
    draw('FNN for logistic regression', train_loss, test_loss)


def mnist_cls(act_func: str, penalty: bool = False):
    data_train, data_test = Dataset.data_mnist()
    net_mnist = NetCls(num_inputs=28 * 28, num_outputs=10, cls_items=10, act=act_func)
    lr = 0.05
    optim = SGD(net_mnist.params, lr)
    train_loss, test_loss = train(net_mnist, data_train, data_test, loss=CrossEntropyLoss(), lr=lr, optimizer=optim,
                                  penalty=penalty)
    draw('FNN for logistic regression', train_loss, test_loss)


def main(args=' ', *kargs):
    print(args, kargs)
    seed_torch()
    # regression()
    # logic_regression()
    mnist_cls('relu')
    # mnist_cls('sigmoid')
    # mnist_cls('elu')
    # mnist_cls('tanh')
    # mnist_cls('relu', penalty=True)


if __name__ == '__main__':
    app.run(main())
