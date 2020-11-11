import torch.nn as nn


class CNN_network(nn.Module):
    def __init__(self):
        super(CNN_network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(28*28, 12)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())  # torch.Size([4, 6, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([4, 16, 8, 8])
        out = out.reshape(out.size(0), -1)
        # print(out.size())  # torch.Size([4, 1024])
        out = self.fc(out)
        return out

