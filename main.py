import CNN_network
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 允许副本存在
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

my_transforms = transforms.Compose([transforms.Grayscale(),  # 转灰度图
                                    # transforms.Resize((32, 32)),  # 重置图像分辨
                                    # transforms.CenterCrop(32),  # 进行中心剪裁
                                    transforms.ToTensor()])  # 转为Tensor

# train_size = "600"  # 每个文字分类取的训练集的个数
train_size = "500"

# train_set_prams
train_set_path = r"./data/" + train_size + "/train"  # 训练集样本的文件夹
train_batch_size = 4  # 训练集的batch_size

# develop_set_prams
develop_set_path = r"./data/" + train_size + "/develop"  # 开发集样本的文件夹
develop_batch_size = 20  # 开发集的batch_size

# test_set_prams
test_set_path = develop_set_path  # 测试集样本的文件夹
test_batch_size = 20  # 测试集的batch_size

# network_save
lab_index = '1'  # 实验编号
net_save_path = 'test/' + lab_index + '/jxw_cnn_net.pkl'  # 训练好的卷积网络存储的地址
params_save_path = 'test/' + lab_index + '/jxw_cnn_net_params.pkl'  # 训练好的卷积网络的参数存储的地址

# 超参数
epoch_count = 20  # 指定的最大epoch数
my_learning_rate = 0.001  # 学习率
my_moment = 0.9  #
my_criterion = nn.CrossEntropyLoss()


def my_optimizer(net):
    # return optim.Adam(net.parameters(), lr=my_learning_rate)
    return optim.SGD(net.parameters(), lr=my_learning_rate, momentum=my_moment)


def load_train_set():
    train_set = torchvision.datasets.ImageFolder(train_set_path, transform=my_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    return train_loader, len(train_set.samples)


def load_develop_set():
    develop_set = torchvision.datasets.ImageFolder(develop_set_path, transform=my_transforms)
    develop_loader = torch.utils.data.DataLoader(develop_set, batch_size=develop_batch_size, shuffle=False)
    return develop_loader, len(develop_set.samples)


def load_test_set():
    test_set = torchvision.datasets.ImageFolder(test_set_path, transform=my_transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return test_loader, len(test_set.samples)


# train
def train_save_net():
    net = CNN_network.CNN_network()
    optimizer = my_optimizer(net)
    criterion = my_criterion
    train_loader, train_sample_size = load_train_set()  # load train set
    develop_loader, develop_sample_size = load_develop_set()  # load develop set

    print('---Data_got---')
    print('---CNN_NET_Train_Start---')

    train_losses = []  # 训练集的loss
    train_right_rates = []  # 训练集的正确率
    develop_losses = []  # 开发集的loss
    develop_right_rates = []  # 开发集的正确率

    if __name__ == '__main__':
        for epoch in range(epoch_count):
            train_loss = 0.0  # 训练集总loss
            train_right_count = 0  # 训练集总正确个数
            for data in train_loader:
                inputs, labels = data
                optimizer.zero_grad()  # 梯度清零，为了不再backward的时候累计梯度
                p_outputs = net(inputs)  # forward，正向传播
                loss = criterion(p_outputs, labels)  # 得到loss
                loss.backward()  # backward，向使得loss减小的梯度下降的方向调整权重，反向传播
                optimizer.step()  # optimizer，很像权重参数
                train_loss += loss.item()  # 累加loss
                # 更新经过这个batch后的到现在为止训练集中正确个数
                train_right_count = update_right_count_by_batch(train_right_count, labels, p_outputs)

            # 测试用当前网络预测开发集中的数据的loss和正确个数
            develop_loss, develop_right_count = test_loss_and_right_count(net, develop_loader, criterion)

            # 计算loss和正确率
            train_loss = train_loss / train_sample_size
            train_right_rate = train_right_count / train_sample_size
            develop_loss = develop_loss / develop_sample_size
            develop_right_rate = develop_right_count / develop_sample_size

            # 存loss和正确率
            train_losses.append(train_loss)
            train_right_rates.append(train_right_rate)
            develop_losses.append(develop_loss)
            develop_right_rates.append(develop_right_rate)

            # 输出结果
            print('epoch[%2d/%2d]' % (epoch + 1, epoch_count))
            print('train train loss=%.5f,\t train right rate=%.3f%%' %
                  (train_loss, train_right_rate * 100))
            print('develop loss=%.5f,\t develop right rate=%.3f%%' %
                  (develop_loss, develop_right_rate * 100))

    print('---CNN_NET_Train_Finished---')

    # 存网络
    torch.save(net, net_save_path)
    torch.save(net.state_dict(), params_save_path)
    print('---CNN_NET_Saved---')

    return train_losses, train_right_rates, develop_losses, develop_right_rates


# 更新一个batch的正确个数
def update_right_count_by_batch(right_count, labels, p_outputs):
    p_outputs_size = p_outputs.size(0)
    predicts = torch.max(p_outputs.data, 1).indices
    # print(torch.max(p_outputs.data, 1))
    for output_index in range(p_outputs_size):
        if predicts[output_index] == labels[output_index]:
            right_count = right_count + 1
    return right_count


# 获取当前loader的总loss和正确个数
def test_loss_and_right_count(net, data_loader, criterion):
    right_count = 0
    test_loss = 0.0
    for data in data_loader:
        inputs, labels = data
        p_outputs = net(inputs)  # forward
        right_count = update_right_count_by_batch(right_count, labels, p_outputs)  # 获得正个数
        test_loss += criterion(p_outputs, labels).item()  # 累加loss
    return test_loss, right_count


# test
def test():
    print('---Test_Start---')

    # 加载网络和测试集
    trained_net = torch.load(net_save_path)  # load net
    test_loader, test_sample_size = load_test_set()  # load test set

    # 测试用当前网络预测测试集中的数据的loss和正确个数
    test_loss, test_right_count = test_loss_and_right_count(trained_net, test_loader,
                                                            my_criterion)
    print('---Test_Finished---')

    # 计算测试集的loss和正确率
    test_loss = test_loss / test_sample_size
    test_right_rate = test_right_count / test_sample_size

    # 打印结果
    print('test loss=%.5f,\t test right rate=%.3f%%' %
          (test_loss, test_right_rate * 100))


# 绘制正确率图
def draw_right_rate_compare(test_right_rates, develop_right_rates):
    plt.figure(2)
    xs = range(1, len(test_right_rates) + 1)
    plt.plot(xs, test_right_rates)
    plt.plot(xs, develop_right_rates, color='red', linestyle='--')
    plt.show()


# 绘制loss图
def draw_loss_compare(train_losses, develop_losses):
    plt.figure(1)
    xs = range(1, len(train_losses) + 1)
    plt.plot(xs, train_losses)
    plt.plot(xs, develop_losses, color='red', linestyle='--')
    plt.show()


# 训练和保存网络
t_ls, t_r_rs, d_ls, d_r_rs = train_save_net()

# 画图
draw_loss_compare(t_ls, d_ls)
draw_right_rate_compare(t_r_rs, d_r_rs)

# test()
