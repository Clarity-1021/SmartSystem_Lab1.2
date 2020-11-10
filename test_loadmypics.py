import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.switch_backend('agg')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        self.fc = nn.Linear(1024, 12)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())  # torch.Size([4, 6, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([4, 16, 8, 8])
        out = out.reshape(out.size(0), -1)
        # print(out.size())  # torch.Size([4, 1024])
        out = self.fc(out)
        return out


running_check_loss_mode = False
my_transforms = transforms.Compose([transforms.Grayscale(),  # 转灰度图
                                    transforms.Resize((32, 32)),  # 重置图像分辨
                                    transforms.CenterCrop(32),  # 进行中心剪裁
                                    transforms.ToTensor()])  # 转为Tensor
train_size = "600"
# train_size = "500"
train_set_path = r"./data/" + train_size + "/train"
develop_set_path = r"./data/" + train_size + "/develop"
test_set_path = develop_set_path
net_save_path = 'jxw_cnn_net.pkl'
params_save_path = 'jxw_cnn_net_params.pkl'
epoch_count = 5
learning = 0.001
moment = 0.9


def load_train_set():
    train_set = torchvision.datasets.ImageFolder(train_set_path, transform=my_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    return train_loader, len(train_set.samples)


def load_develop_set():
    develop_set = torchvision.datasets.ImageFolder(develop_set_path, transform=my_transforms)
    develop_loader = torch.utils.data.DataLoader(develop_set, batch_size=4, shuffle=True)
    return develop_loader, len(develop_set.samples)


def load_test_set():
    test_set = torchvision.datasets.ImageFolder(test_set_path, transform=my_transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True)
    return test_loader, len(test_set.samples)


classes = ('博', '学', '笃', '志', '切', '问', '近', '思', '自', '由', '无', '用')


# train
def train_save_net():
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=learning, momentum=moment)
    criterion = nn.CrossEntropyLoss()
    train_loader, train_sample_size = load_train_set()
    develop_loader, develop_sample_size = load_develop_set()

    print('---Data_got---')

    if __name__ == '__main__':
        for epoch in range(epoch_count):
            print('epoch[%2d/%2d]' % (epoch + 1, epoch_count))

            train_loss = 0.0
            running_loss = 0.0
            develop_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if running_check_loss_mode:
                    running_loss += loss.item()
                    if i % 200 == 199:
                        for j, data in enumerate(develop_loader, 0):
                            # get the inputs
                            inputs, labels = data

                            outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            optimizer.step()
                            develop_loss += loss.item()

                        print('[%d]' % (i + 1))
                        print('running train loss=%.5f, develop loss=%.5f' %
                              (running_loss / 200, develop_loss / develop_sample_size))
                        running_loss = 0.0
                        develop_loss = 0.0
                else:
                    train_loss += loss.item()

            if not running_check_loss_mode:
                for k, data in enumerate(develop_loader, 0):
                    # get the inputs
                    inputs, labels = data

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    optimizer.step()
                    develop_loss += loss.item()

                print('train loss=%.5f, develop loss=%.5f' %
                      (train_loss / train_sample_size, develop_loss / develop_sample_size))

    print('---CNN_NET_Trained---')

    torch.save(net, net_save_path)
    torch.save(net.state_dict(), params_save_path)

    print('---CNN_NET_Saved---')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# test
def test():
    trained_net = torch.load(net_save_path)  # load net
    test_loader, test_sample_size = load_test_set()  # load test data set
    data_iter = iter(test_loader)
    images, labels = data_iter.next()  #
    # imshow(torchvision.utils.make_grid(images, nrow=5))
    print('GroundTruth: ', " ".join('%5s' % classes[labels[j]] for j in range(2)))
    outputs = trained_net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(2)))


train_save_net()
test()
