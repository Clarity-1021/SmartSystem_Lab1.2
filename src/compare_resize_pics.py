from src import CNN_network
from src import net_32
from src import load_data
from src import load_data_32
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
lals = ['28x28', '32x32']
# 用相同的optimiser训练网络，配合不同的损失函数

net_28 = CNN_network.CNN_network()
net_32 = net_32.CNN_network()
# nets = [net_28, net_32]

opt_28 = optim.Adamax(net_28.parameters(),lr=my_learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
opt_32 = optim.Adamax(net_32.parameters(),lr=my_learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
# opts = [opt_28, opt_32]

criterion = nn.CrossEntropyLoss()

train_loader_28, train_sample_size_28, train_batch_size_28 = load_data.load_train_set()  # load train set
develop_loader_28, develop_sample_size_28, develop_batch_size_28 = load_data.load_develop_set()  # load develop set
train_loader_32, train_sample_size_32, train_batch_size_32 = load_data_32.load_train_set()  # load train set
develop_loader_32, develop_sample_size_32, develop_batch_size_32 = load_data_32.load_develop_set()  # load develop set

print('---Data_got---')
print('---CNN_NET_Train_Start---')

train_losses_cps = [[], []]  # 训练集的loss
train_right_rates_cps = [[], []]  # 正确率
develop_losses_cps = [[], []]  # 训练集的loss
develop_right_rates_cps = [[], []]

def test_loss_and_right_rate_28():
    test_loss = 0.0  # 总loss
    test_right_count = 0  # 总正确个数
    for data in develop_loader_28:
        inputs, labels = data
        p_outputs = net_28(inputs)  # forward，正向传播
        loss = criterion(p_outputs, labels)
        test_loss += loss.item()
        test_right_count += torch.eq(torch.max(p_outputs.data, 1).indices, labels).sum().item()

    print('---develop_set---')
    tr_ll = test_loss / len(develop_loader_28)
    tr_rr = test_right_count / develop_sample_size_28
    develop_losses_cps[0].append(tr_ll)
    develop_right_rates_cps[0].append(tr_rr)
    print('[28x28] develop loss=%.5f,\t develop right rate=%.3f%%' %
          (tr_ll, tr_rr * 100))


def test_loss_and_right_rate_32():
    test_loss = 0.0  # 总loss
    test_right_count = 0  # 总正确个数
    for data in develop_loader_32:
        inputs, labels = data
        p_outputs = net_32(inputs)  # forward，正向传播
        loss = criterion(p_outputs, labels)
        test_loss += loss.item()
        test_right_count += torch.eq(torch.max(p_outputs.data, 1).indices, labels).sum().item()

    print('---develop_set---')
    tr_ll = test_loss / len(develop_loader_32)
    tr_rr = test_right_count / develop_sample_size_32
    develop_losses_cps[1].append(tr_ll)
    develop_right_rates_cps[1].append(tr_rr)
    print('[32x32] develop loss=%.5f,\t develop right rate=%.3f%%' %
          (tr_ll, tr_rr * 100))



if __name__ == '__main__':
    for epoch in range(epoch_count):
        train_loss = 0.0  # 总loss
        train_right_count = 0  # 总正确个数
        for batch_index, data in enumerate(train_loader_28):
            inputs, labels = data
            opt_28.zero_grad()  # 梯度清零，为了不再backward的时候累计梯度
            p_outputs = net_28(inputs)  # forward，正向传播
            loss = criterion(p_outputs, labels)
            loss.backward()  # backward，向使得loss减小的梯度下降的方向调整权重，反向传播
            opt_28.step()
            train_loss += loss.item()
            train_right_count += torch.eq(torch.max(p_outputs.data, 1).indices, labels).sum().item()
            if batch_index % 2400 == 2399:
                print('epoch[%2d/%2d] batch[%4d/%4d]' % (epoch + 1, epoch_count, batch_index + 1, len(train_loader_28)))
                print('---train_set---')
                tr_ll = train_loss / 2400
                tr_rr = train_right_count / (2400 * train_batch_size_28)
                train_losses_cps[0].append(tr_ll)
                train_right_rates_cps[0].append(tr_rr)
                print('[28x28] train loss=%.5f,\t train right rate=%.3f%%' %
                      (tr_ll, tr_rr * 100))
                train_loss = 0.0  # 总loss
                train_right_count = 0  # 总正确个数
                test_loss_and_right_rate_28()

        train_loss = 0.0  # 总loss
        train_right_count = 0  # 总正确个数
        for batch_index, data in enumerate(train_loader_32):
            inputs, labels = data
            opt_32.zero_grad()  # 梯度清零，为了不再backward的时候累计梯度
            p_outputs = net_32(inputs)  # forward，正向传播
            loss = criterion(p_outputs, labels)
            loss.backward()  # backward，向使得loss减小的梯度下降的方向调整权重，反向传播
            opt_32.step()
            train_loss += loss.item()
            train_right_count += torch.eq(torch.max(p_outputs.data, 1).indices, labels).sum().item()
            if batch_index % 2400 == 2399:
                print('epoch[%2d/%2d] batch[%4d/%4d]' % (epoch + 1, epoch_count, batch_index + 1, len(train_loader_32)))
                print('---train_set---')
                tr_ll = train_loss / 2400
                tr_rr = train_right_count / (2400 * train_batch_size_32)
                train_losses_cps[1].append(tr_ll)
                train_right_rates_cps[1].append(tr_rr)
                print('[32x32] train loss=%.5f,\t train right rate=%.3f%%' %
                      (tr_ll, tr_rr * 100))
                train_loss = 0.0  # 总loss
                train_right_count = 0  # 总正确个数
                test_loss_and_right_rate_32()

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



