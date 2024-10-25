import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from .net import Net
import time


def evaluate_accuracy(data_test, model: Net, loss=CrossEntropyLoss()):
    acc_sum, n = 0.0, 0
    test_l_sum = 0.0
    for X, y in data_test:
        acc_sum += (model.forward(X).argmax(dim=1) == y).float().sum().item()
        l = loss(model.forward(X).squeeze(-1), y).sum()
        test_l_sum += l.item()
        n += y.shape[0]
    return acc_sum / n, test_l_sum / n


def l2_penalty(w1, w2):
    return ((w1 ** 2).sum() + (w2 ** 2).sum()) / 2


def train(model: Net, data_train, data_test,
          loss=CrossEntropyLoss,
          num_epochs=50,
          params=None,
          lr=0.05,
          optimizer=None,
          penalty=False,
          lambd=1):

    train_loss = []
    test_loss = []
    begintime = time.time()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in data_train:
            y_hat = model.forward(X)
            l = loss(y_hat.squeeze(-1), y)
            if penalty:
                l += lambd * l2_penalty(model.params[0], model.params[2])
            l = l.sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                SGD(params, lr)
            else:
                optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc, test_l = evaluate_accuracy(data_test, model, loss)
        train_loss.append(train_l_sum / n)
        test_loss.append(test_l)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    endtime = time.time()
    print("FNN-regression : %d epochs total time: %.3fs" % (num_epochs, endtime - begintime))
    return train_loss, test_loss
