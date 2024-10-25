import numpy as np
import torch
import torch.nn as nn


class Net():
    def __init__(self, num_inputs=500, num_hiddens=256, num_outputs=1, act: str = 'relu'):
        w_1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float32,
                           requires_grad=True)
        b_1 = torch.zeros(num_hiddens, dtype=torch.float32, requires_grad=True)
        w_2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float32,
                           requires_grad=True)
        b_2 = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)
        self.params = [w_1, b_1, w_2, b_2]

        # 定义模型结构
        self.input_layer = lambda x: x.view(x.shape[0], -1)
        self.hidden_layer = lambda x: self.activate_func(torch.matmul(x, w_1.t()) + b_1)
        self.output_layer = lambda x: torch.matmul(x, w_2.t()) + b_2

        self._act = None
        self.activate_func(act)

    def activate_func(self, act: str = 'relu'):
        act_func = None
        try:
            if act == 'relu':
                act_func = nn.ReLU()
            elif act == 'sigmoid':
                act_func = nn.Sigmoid()
            elif act == 'tanh':
                act_func = nn.Tanh()
            elif act == 'elu':
                act_func = nn.ELU()
            self._act = act_func
        except Exception as e:
            print(f"act should be one of relu,sigmoid,tanh,elu\n{e}")
            raise

    def forward(self, x):
        x = self.input_layer(x)
        x = self._act(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def __call__(self, x):
        self.forward(x)


class NetCls(Net):
    """
    default classification items = 2
    """

    def __init__(self, num_inputs=200, num_hiddens=256, num_outputs=1, cls_items=2, ):
        super().__init__(num_inputs, num_hiddens, num_outputs)
        self.cls_items = cls_items
        self.fn_logistic = self.logistic

    def logistic(self, x):
        x = 1.0 / (1.0 + torch.exp(-x))
        return x

    def forward(self, x):
        super().forward(x)
        if self.cls_items == 2:
            x = self.fn_logistic(x)
        return x

    def __call__(self, x):
        self.forward(x)
