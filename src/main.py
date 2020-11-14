from src import CNN_network
from src import load_data
from src import tools
import torch
import torch.nn as nn
import torch.optim as optim

# network_save
lab_index = '5'  # 实验编号
net_save_name = '../lab_result/' + lab_index + '/jxw_cnn_net'  # 训练好的卷积网络
params_save_name = '../lab_result/' + lab_index + '/jxw_cnn_net_params'  # 训练好的卷积网络的参数

# 超参数
epoch_count = 10  # 指定的最大epoch数
my_learning_rate = 0.01  # 学习率
my_criterion = nn.CrossEntropyLoss()
batch_count = 2400

# train
net = CNN_network.CNN_network()
optimizer = optim.Adamax(net.parameters(),lr=my_learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
criterion = my_criterion
train_loader, train_sample_size, train_batch_size = load_data.load_train_set()  # load train set
develop_loader, develop_sample_size, develop_batch_size = load_data.load_develop_set()  # load develop set

print('---Data_got---')
print('---CNN_NET_Train_Start---')

train_losses = []  # 训练集的loss
train_right_rates = []  # 训练集的正确率
develop_losses = []  # 开发集的loss
develop_right_rates = []  # 开发集的正确率
max_develop_right_rate = 0.0
isStop = False

if __name__ == '__main__':
    for epoch in range(epoch_count):
        if isStop:
            break
        train_loss = 0.0  # 训练集总loss
        train_right_count = 0  # 训练集总正确个数
        for batch_index, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()  # 梯度清零，为了不再backward的时候累计梯度
            p_outputs = net(inputs)  # forward，正向传播
            loss = criterion(p_outputs, labels)  # 得到loss
            loss.backward()  # backward，向使得loss减小的梯度下降的方向调整权重，反向传播
            optimizer.step()  # optimizer，更新权重参数
            train_loss += loss.item()  # 累加loss
            # 更新经过这个batch后的到现在为止训练集中正确个数
            train_right_count += torch.eq(torch.max(p_outputs.data, 1).indices, labels).sum().item()
            # train_right_count = update_right_count_by_batch(train_right_count, labels, p_outputs)

            if batch_index % batch_count == (batch_count - 1):
                # 测试用当前网络预测开发集中的数据的loss和正确个数
                develop_loss, develop_right_count = tools.test_loss_and_right_count(net, develop_loader, criterion)

                # 计算loss和正确率
                train_loss = train_loss / batch_count
                train_right_rate = train_right_count / (batch_count * train_batch_size)
                develop_loss = develop_loss / len(develop_loader)
                develop_right_rate = develop_right_count / develop_sample_size

                # 存loss和正确率用来画图
                train_losses.append(train_loss)
                train_right_rates.append(train_right_rate)
                develop_losses.append(develop_loss)
                develop_right_rates.append(develop_right_rate)

                # 输出结果
                print('epoch[%2d/%2d] batch[%4d/%4d]' % (epoch + 1, epoch_count, batch_index + 1, len(train_loader)))
                print('[train]\t loss=%.5f,\t right rate=%.3f%%' %
                      (train_loss, train_right_rate * 100))
                print('[develop]\t loss=%.5f,\t right rate=%.3f%%' %
                      (develop_loss, develop_right_rate * 100))

                # 正确率高于之前的峰值，保存网络
                if develop_right_rate >= max_develop_right_rate:
                    max_develop_right_rate = develop_right_rate
                    suffix = '_%d_%d_%.3f%%' % (epoch + 1, batch_index + 1, develop_right_rate * 100) + '.pkl'
                    net_save_path = net_save_name + suffix
                    params_save_path = params_save_name + suffix
                    torch.save(net, net_save_path)
                    torch.save(net.state_dict(), params_save_path)

                if train_loss < 0.001:
                    isStop = True
                    break

                train_loss = 0.0  # 训练集总loss
                train_right_count = 0  # 训练集总正确个数

print('---CNN_NET_Train_Finished---')

# 画图
tools.draw_loss_compare(train_losses, develop_losses)
tools.draw_right_rate_compare(train_right_rates, develop_right_rates)

