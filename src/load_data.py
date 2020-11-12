import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


my_transforms = transforms.Compose([transforms.Grayscale(),  # 转灰度图
                                    # transforms.Resize((32, 32)),  # 重置图像分辨
                                    # transforms.CenterCrop(32),  # 进行中心剪裁
                                    transforms.ToTensor()])  # 转为Tensor

train_size = "600"  # 每个文字分类取的训练集的个数
# train_size = "500"

# train_set_prams
train_set_path = r"../data/" + train_size + "/train"  # 训练集样本的文件夹
train_batch_size = 5  # 训练集的batch_size

# develop_set_prams
develop_set_path = r"../data/" + train_size + "/develop"  # 开发集样本的文件夹
develop_batch_size = 20  # 开发集的batch_size

# test_set_prams
test_set_path = develop_set_path  # 测试集样本的文件夹
test_batch_size = 20  # 测试集的batch_size


def load_train_set():
    train_set = torchvision.datasets.ImageFolder(train_set_path, transform=my_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    return train_loader, len(train_set.samples), train_batch_size


def load_develop_set():
    develop_set = torchvision.datasets.ImageFolder(develop_set_path, transform=my_transforms)
    develop_loader = torch.utils.data.DataLoader(develop_set, batch_size=develop_batch_size, shuffle=False)
    return develop_loader, len(develop_set.samples), develop_batch_size


def load_test_set():
    test_set = torchvision.datasets.ImageFolder(test_set_path, transform=my_transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return test_loader, len(test_set.samples), test_batch_size