import torch.nn as nn


# 其他的激活函数。。。不咋需要
# torch.nn.ReLU类属于非线性激活分类，在定义时默认不需要传入参数。
# 当然，在 torch.nn包中还有许多非线性激活函数类可供选择，
# 比如之前讲到的PReLU、LeakyReLU、Tanh、Sigmoid、Softmax等。
class CNN_network(nn.Module):
    def __init__(self):
        super(CNN_network, self).__init__()
        # 第一个隐藏层
        # nn.Sequential是一个有序的容器，按照构造器中的顺序依次被添加到计算图中执行
        self.layer1 = nn.Sequential(
            # 从输入层到第一层隐藏层
            # input channels = 1 因为是灰度图，输入只有一维，”0“or"1"
            # output channels = 6 同时也是第二层的input channels数
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            # 激活函数，等效为“max(features, 0)”，如果是正保持不变，小于"0"置0
            nn.ReLU(),
            # 池化，窗口大小为2x2，即从这四个值里挑一个最大的作为输出
            # 移动的窗口移动的步长为2
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二个隐藏层
        self.layer2 = nn.Sequential(
            # 从第一个隐藏层到第二个隐藏层
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 从最后一个隐藏层到输出层的线性变化
        self.fc = nn.Linear(32*32, 12)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())  # torch.Size([4, 6, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([4, 16, 8, 8])
        out = out.reshape(out.size(0), -1)
        # print(out.size())  # torch.Size([4, 28*28])
        out = self.fc(out)
        return out