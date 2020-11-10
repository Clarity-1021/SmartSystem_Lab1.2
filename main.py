import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class CNN_network(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_network, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, 10, 3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        # self.conv2 = nn.Conv2d(10, out_channels, 3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.pool1(out)
        # out = self.conv2(out)
        # out = self.pool2(out)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

#
# # 设备配置
# torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 超参数设置
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# print(device)

# 训练数据集
path = r"./train"
train_dataset = torchvision.datasets.ImageFolder(path,
                                                 transform=transforms.Compose([
                                                     transforms.Resize((128, 128)),
                                                     transforms.ToTensor()]))

# 测试数据集
test_dataset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((128, 128)),
                                                    transforms.ToTensor()]))

# 数据加载器
# 训练数据 加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 实例化一个模型，并迁移至gpu
model = CNN_network(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
if __name__ == '__main__':
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 注意模型在GPU中，数据也要搬到GPU中
            images = images
            labels = labels

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

