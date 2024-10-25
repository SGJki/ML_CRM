from utils.dataset import Dataset
from utils.train import train
from utils.draw import draw
from utils.net import Net, NetCls


def regression():
    data_train, data_test = Dataset.data_reg()
    net_reg = Net()
    train_loss, test_loss = train(net_reg, data_train, data_test)
    draw('FNN for regression', train_loss, test_loss)



def logic_regression():
    pass


def mnist_cls(act_func: str, penalty=False):
    pass


def main():
    regression()
    # logic_regression()
    # mnist_cls('relu')
    # mnist_cls('sigmoid')
    # mnist_cls('elu')
    # mnist_cls('tanh')
    # mnist_cls('relu', penalty=True)

if __name__ == '__main__':
    main()