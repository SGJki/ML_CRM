import numpy as np
from sklearn import datasets
import mindspore as ms
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.callback import Callback
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as transforms
import random

class MyDataSet(object):
    def __init__(self, input, label, istart=0, iend=150):
        super(MyDataSet, self).__init__()
        self.x = input[istart:iend, :]
        self.y = label[istart:iend]
        self.len = self.x.shape[0]
        to_onehot = transforms.OneHot(num_classes=3)
        self.y = to_onehot(self.y)
        self.y = np.array(self.y,dtype=np.float32)
    def __len__(self):
        return self.len
    def __getitem__(self, item):
        x,y = self.x[item,:],self.y[item,:]
        return x,y
class MyNN(nn.Cell):
    def __init__(self, n_feature, n_output):
        super(MyNN, self).__init__()
        self.f1 = nn.Dense(n_feature, 128)
        self.f2 = nn.Dense(128, 64)
        self.f3 = nn.Dense(64, 32)
        self.f4 = nn.Dense(32, n_output)
        self.nl = nn.ReLU()
    def construct(self, x):
        x = self.nl(self.f1(x))
        x = self.nl(self.f2(x))
        x = self.nl(self.f3(x))
        x = self.f4(x)
        return x
class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=True)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(acc["accuracy"])
            self.epoch_per_eval["loss"].append(acc["loss"])
            print(acc)
class TrainLossCallBack(Callback):
    def __init__(self):
        super(TrainLossCallBack, self).__init__()
        self.train_epoch_loss = []
        self.epoch_per_train = []
    def epoch_begin(self, run_context):
        self.losses = []
    def epoch_end(self, run_context):
        callback_params = run_context.original_args()
        self.epoch_per_train.append(callback_params.cur_epoch_num)
        self.train_epoch_loss.append(np.mean(self.losses))
        print(f"epoch : {callback_params.cur_epoch_num}, "
              f"avg loss: {np.mean(self.losses):5.3f}", flush=True)
    def step_end(self, run_context):
        callback_params = run_context.original_args()
        loss = callback_params.net_outputs
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarry):
                loss = loss[0]
        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        self.losses.append(loss)

if __name__ == '__main__':
    dataset = datasets.load_iris()
    data_raw = dataset['data']
    target_raw = dataset['target']
    #打乱顺序
    id = list(range(0,150)); random.shuffle(id)
    data, target = [], []
    for i in range(0,150):
        data.append(data_raw[id[i]])
        target.append(target_raw[id[i]])
    # print(target_raw)
    # print(target)
    #类型转换和归一化
    input = np.array(data, dtype=np.float32)
    input = (input-input.min(axis=0))/(input.max(axis=0)-input.min(axis=0))
    label = np.array(target)
    # # 训练集
    dset_train_raw = MyDataSet(input, label, istart=0, iend=120)
    dset_train = GeneratorDataset(source=dset_train_raw, column_names=["x", "y"], shuffle=True)
    dset_train = dset_train.batch(10)
    # 测试集
    dset_test_raw = MyDataSet(input, label, istart=120, iend=150)
    dset_test = GeneratorDataset(source=dset_test_raw, column_names=["x", "y"], shuffle=True)
    dset_test = dset_test.batch(10)
    # 模型
    net = MyNN(n_feature=4, n_output=3)
    # for c in net.cells_and_names():
    #     print(c)
    net_opt = nn.Adam(net.trainable_params(), learning_rate=0.01, weight_decay=0.1)
    net_loss = nn.SoftmaxCrossEntropyWithLogits()
    model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy', 'loss'})
    ts_cb = TrainLossCallBack()
    epoch_per_eval = {"epoch": [], "acc": [], "loss": []}
    eval_per_epoch = 1
    eval_cb = EvalCallBack(model, dset_test, eval_per_epoch, epoch_per_eval)
    model.train(100, dset_train, callbacks=[eval_cb,ts_cb])
    # 准确率
    accm = max(epoch_per_eval["acc"])
    print(accm,epoch_per_eval["epoch"][epoch_per_eval["acc"].index(accm)])

    # 绘制曲线
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(epoch_per_eval["epoch"], epoch_per_eval["loss"], 'r-', label=r'loss')
    plt.legend()
    plt.xlabel(r'epoch number')
    plt.ylabel(r'test loss')
    plt.twinx()
    plt.plot(epoch_per_eval["epoch"], epoch_per_eval["acc"], 'b-', label=r'accuracy')
    plt.ylabel(r'test accuracy')
    plt.legend()
    plt.figure()
    plt.title("train loss")
    plt.xlabel("epoch number")
    plt.ylabel("Model Loss")
    plt.plot(ts_cb.epoch_per_train, ts_cb.train_epoch_loss, "red")
    plt.show()


