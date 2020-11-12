from torch.autograd import Variable
from src import CNN_network
from src import load_data
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 允许副本存在
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 超参数
epoch_count = 20  # 指定的最大epoch数
my_learning_rate = 0.001  # 学习率
my_moment = 0.9


def my_optimizer(net):
    # return optim.Adam(net.parameters(), lr=my_learning_rate)
    return optim.SGD(net.parameters(), lr=my_learning_rate, momentum=my_moment)


# train
lals = ['MSELoss', 'L1Loss', 'CrossEntropyLoss']
# 用相同的optimiser训练网络，配合不同的损失函数
net_MSELoss = CNN_network.CNN_network()
net_L1Loss = CNN_network.CNN_network()
net_CrossEntropyLoss = CNN_network.CNN_network()
nets = [net_MSELoss, net_L1Loss, net_CrossEntropyLoss]

opt_MSELoss = my_optimizer(net_MSELoss)
opt_L1Loss = my_optimizer(net_L1Loss)
opt_CrossEntropyLoss = my_optimizer(net_CrossEntropyLoss)
opts = [opt_MSELoss, opt_L1Loss, opt_CrossEntropyLoss]

crts = [nn.MSELoss(), nn.L1Loss(), nn.CrossEntropyLoss()]

train_loader, train_sample_size, train_batch_size = load_data.load_train_set()  # load train set

print('---Data_got---')
print('---CNN_NET_Train_Start---')

train_losses_cps = [[], [], []]  # 训练集的loss
train_right_rates_cps = [[], [], []]  # 正确率

if __name__ == '__main__':
    for epoch in range(epoch_count):
        print('epoch[%2d/%2d]' % (epoch + 1, epoch_count))
        train_loss = [[0.0], [0.0], [0.0]]  # 训练集总loss
        train_right_count = [[0], [0], [0]]  # 训练集总正确个数
        for data in train_loader:
            inputs, labels = data
            for net, optimizer, criterion, tr_l, tr_r in zip(
                 nets, opts, crts, train_loss, train_right_count):
                optimizer.zero_grad()  # 梯度清零，为了不再backward的时候累计梯度
                p_outputs = net(inputs)  # forward，正向传播
                if criterion != crts[2]:
                    p_outputs = torch.max(p_outputs.data, 1).indices
                    p_outputs = torch.zeros(train_batch_size, 12).scatter_(1, p_outputs.view(train_batch_size, 1), 1)
                    p_outputs = Variable(p_outputs, requires_grad=True)
                    labels = torch.zeros(train_batch_size, 12).scatter_(1, labels.view(train_batch_size, 1), 1)
                    labels = Variable(labels, requires_grad=True)
                loss = criterion(p_outputs, labels)
                loss.backward()  # backward，向使得loss减小的梯度下降的方向调整权重，反向传播
                optimizer.step()
                tr_l[0] += loss.item()
                p_outputs = torch.max(p_outputs.data, 1).indices
                if criterion != crts[2]:
                    labels = torch.max(labels.data, 1).indices
                tr_r[0] += torch.eq(p_outputs, labels).sum().item()

        for loss_fn, tr_l, tr_r, train_losses, train_right_rates in zip(
                lals, train_loss, train_right_count, train_losses_cps, train_right_rates_cps):
            tr_ll = tr_l[0] / len(train_loader)
            tr_rr = tr_r[0] / train_sample_size
            train_losses.append(tr_ll)
            train_right_rates.append(tr_rr)
            print('['+loss_fn+'] train loss=%.5f,\t train right rate=%.3f%%' %
                  (tr_ll, tr_rr * 100))

print('---CNN_NET_Train_Finished---')

plt.figure(1)
for i, train_losses in enumerate(train_losses_cps):
    plt.plot(train_losses, label=lals[i])
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.figure(2)
for i, train_right_rates in enumerate(train_right_rates_cps):
    plt.plot(train_right_rates, label=lals[i])
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('right rate')
plt.show()



