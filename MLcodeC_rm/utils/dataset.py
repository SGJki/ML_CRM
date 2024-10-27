import numpy as np
import mpmath
import torch
import pathlib
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Dataset_logistic(Dataset):
    def __init__(self, num_samples=10000, feature_dim=200, mean=0.5, std_dev=1.0, label=0, train=True):
        self.num_samples = 7000 if train else 3000
        self.feature_dim = feature_dim
        self.train = train

        # 生成特征
        self.data = np.random.normal(mean, std_dev, (self.num_samples, 1, feature_dim)).astype(np.float32)

        # 生成标签
        self.targets = np.full(self.num_samples, label, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, label

class Dataset(object):
    """
    cls : class of dataset, regression | two-classification | mnist
    """
    batch_size = 128
    ratio = 0.7
    train_dataloader = None
    test_dataloader = None

    @classmethod
    def data_reg(cls, num=10000, dim=500, a=0.0056, b=0.028):
        train_num = int(num * cls.ratio)
        features = torch.randn(num, dim)
        a_mat = a * torch.ones(dim, 1)
        # add normal noise
        labels = torch.matmul(features, a_mat)+ b
        labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
        train_features, test_features = features[:train_num, :], features[train_num:, :]
        train_labels, test_labels = labels[:train_num], labels[train_num:]
        train_dataset1 = TensorDataset(train_features, train_labels)
        test_dataset1 = TensorDataset(test_features, test_labels)
        train_dataloader = DataLoader(dataset=train_dataset1, batch_size=cls.batch_size,
                                                            shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset1, batch_size=cls.batch_size,
                                                           shuffle=True)
        return train_dataloader, test_dataloader

    @classmethod
    def data_cls(cls, num=10000, dim=200, mean=0.2, std=2):
        train_num = int(num * cls.ratio)
        features1 = torch.normal(mean, std, size=(num, dim), dtype=torch.float32)
        labels1 = torch.ones(num)
        features2 = torch.normal(-mean, std, size=(num, dim), dtype=torch.float32)
        labels2 = torch.zeros(num)

        train_features = torch.cat((features1[:train_num], features2[:train_num]),
                                   dim=0)  # size torch.Size([14000, 200])
        train_labels = torch.cat((labels1[:train_num], labels2[:train_num]), dim=-1)  # size  torch.Size([6000, 200])
        test_features = torch.cat((features1[train_num:], features2[train_num:]), dim=0)  # torch.Size([14000])
        test_labels = torch.cat((labels1[train_num:], labels2[train_num:]), dim=-1)  # torch.Size([6000])
        # Build the training and testing dataset
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cls.batch_size,
                                                            shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cls.batch_size,
                                                           shuffle=True)
        return train_dataloader, test_dataloader

    @classmethod
    def data_mnist(cls):
        path = pathlib.Path(__file__).parent
        train_path = path.joinpath("Datasets", "MNIST", "Train")
        test_path = path.joinpath("Datasets", "MNIST", "Test")

        train_dataset = torchvision.datasets.MNIST(root=str(train_path), train=True, download=True,
                                                   transform=transforms.ToTensor())
        test_dataset = torchvision.datasets.MNIST(root=str(test_path), train=False, download=True,
                                                  transform=transforms.ToTensor())
        train_dataloader = DataLoader(train_dataset, batch_size=cls.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=cls.batch_size, shuffle=False)
        return train_dataloader, test_dataloader

    @classmethod
    def data_logistic(cls, num=10000, ratio=0.7, dim=200, std=1.0, label=0, train=True):
        # 创建两个数据集
        train_dataset_0 = Dataset_logistic(mean=0.5, label=0, train=True)
        train_dataset_1 = Dataset_logistic(mean=-0.5, label=1, train=True)
        test_dataset_0 = Dataset_logistic(mean=0.5, label=0, train=False)
        test_dataset_1 = Dataset_logistic(mean=-0.5, label=1, train=False)

        # 合并数据集
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_0, train_dataset_1])
        test_dataset = torch.utils.data.ConcatDataset([test_dataset_0, test_dataset_1])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        return train_loader, test_loader


def main():
    a, _ = Dataset.data_mnist()
    print(a)


if __name__ == "__main__":
    main()
