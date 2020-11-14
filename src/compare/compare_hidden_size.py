from src import CNN_network
from src.compare import net_hidden_1_16, net_hidden_1_6_2_16_3_32, net_hidden_1_6_2_10_3_16, net_hidden_1_6
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
epoch_count = 10  # 指定的最大epoch数
my_learning_rate = 0.01  # 学习率

# train
lals = ['1x6-2x16', '1x6', '1x16', '1x6-2x10-3x16', '1x6-2x16-3x32']
# 用相同的optimiser训练网络，配合不同的损失函数

net_hidden_1_6_2_16 = CNN_network.CNN_network()
net_hidden_1_6 = net_hidden_1_6.CNN_network()
net_hidden_1_16 = net_hidden_1_16.CNN_network()
net_hidden_1_6_2_10_3_16 = net_hidden_1_6_2_10_3_16.CNN_network()
net_hidden_1_6_2_16_3_32 = net_hidden_1_6_2_16_3_32.CNN_network()
nets = [net_hidden_1_6_2_16, net_hidden_1_6, net_hidden_1_16, net_hidden_1_6_2_10_3_16, net_hidden_1_6_2_16_3_32]

opt_hidden_1_6_2_16 = optim.Adamax(net_hidden_1_6_2_16.parameters(),lr=my_learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
opt_hidden_1_6 = optim.Adamax(net_hidden_1_6.parameters(),lr=my_learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
opt_hidden_1_16 = optim.Adamax(net_hidden_1_16.parameters(),lr=my_learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
opt_hidden_1_6_2_10_3_16 = optim.Adamax(net_hidden_1_6_2_10_3_16.parameters(),lr=my_learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
opt_hidden_1_6_2_16_3_32 = optim.Adamax(net_hidden_1_6_2_16_3_32.parameters(),lr=my_learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
opts = [opt_hidden_1_6_2_16, opt_hidden_1_6, opt_hidden_1_16, opt_hidden_1_6_2_10_3_16, opt_hidden_1_6_2_16_3_32]

criterion = nn.CrossEntropyLoss()

train_loader, train_sample_size, train_batch_size = load_data.load_train_set()  # load train set
develop_loader, develop_sample_size, develop_batch_size = load_data.load_develop_set()  # load develop set

print('---Data_got---')
print('---CNN_NET_Train_Start---')

train_losses_cps = [[], [], [], [], []]  # 训练集的loss
train_right_rates_cps = [[], [], [], [], []]  # 正确率
develop_losses_cps = [[], [], [], [], []]  # 训练集的loss
develop_right_rates_cps = [[], [], [], [], []]

def test_loss_and_right_rate():
    test_loss = [[0.0], [0.0], [0.0], [0.0], [0.0]]  # 总loss
    test_right_count = [[0], [0], [0], [0], [0]]  # 总正确个数
    for data in develop_loader:
        inputs, labels = data
        for net, tr_l, tr_r in zip(nets, test_loss, test_right_count):
            p_outputs = net(inputs)  # forward，正向传播
            loss = criterion(p_outputs, labels)
            tr_l[0] += loss.item()
            tr_r[0] += torch.eq(torch.max(p_outputs.data, 1).indices, labels).sum().item()

    print('---develop_set---')
    for optm, tr_l, tr_r, develop_losses, develop_right_rates in zip(
         lals, test_loss, test_right_count, develop_losses_cps, develop_right_rates_cps):
        tr_ll = tr_l[0] / len(develop_loader)
        tr_rr = tr_r[0] / develop_sample_size
        develop_losses.append(tr_ll)
        develop_right_rates.append(tr_rr)
        print('['+optm+'] develop loss=%.5f,\t develop right rate=%.3f%%' %
              (tr_ll, tr_rr * 100))


if __name__ == '__main__':
    for epoch in range(epoch_count):
        train_loss = [[0.0], [0.0], [0.0], [0.0], [0.0]]  # 总loss
        train_right_count = [[0], [0], [0], [0], [0]]  # 总正确个数
        for batch_index, data in enumerate(train_loader):
            inputs, labels = data
            for net, optimizer, tr_l, tr_r in zip(
                 nets, opts, train_loss, train_right_count):
                optimizer.zero_grad()  # 梯度清零，为了不再backward的时候累计梯度
                p_outputs = net(inputs)  # forward，正向传播
                loss = criterion(p_outputs, labels)
                loss.backward()  # backward，向使得loss减小的梯度下降的方向调整权重，反向传播
                optimizer.step()
                tr_l[0] += loss.item()
                tr_r[0] += torch.eq(torch.max(p_outputs.data, 1).indices, labels).sum().item()
            if batch_index % 2400 == 2399:
                print('epoch[%2d/%2d] batch[%4d/%4d]' % (epoch + 1, epoch_count, batch_index + 1, len(train_loader)))
                print('---train_set---')
                for optm, tr_l, tr_r, train_losses, train_right_rates in zip(
                        lals, train_loss, train_right_count, train_losses_cps, train_right_rates_cps):
                    tr_ll = tr_l[0] / 2400
                    tr_rr = tr_r[0] / (2400 * train_batch_size)
                    train_losses.append(tr_ll)
                    train_right_rates.append(tr_rr)
                    print('[' + optm + '] train loss=%.5f,\t train right rate=%.3f%%' %
                          (tr_ll, tr_rr * 100))
                train_loss = [[0.0], [0.0], [0.0], [0.0], [0.0]]  # 总loss
                train_right_count = [[0], [0], [0], [0], [0]]  # 总正确个数
                test_loss_and_right_rate()

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

plt.figure(3)
for i, develop_losses in enumerate(develop_losses_cps):
    plt.plot(develop_losses, label=lals[i])
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.figure(4)
for i, develop_right_rates in enumerate(develop_right_rates_cps):
    plt.plot(develop_right_rates, label=lals[i])
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('right rate')
plt.show()





