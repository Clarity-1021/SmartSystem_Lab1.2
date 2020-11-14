import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 允许副本存在
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 获取当前loader的总loss和正确个数
def test_loss_and_right_count(net, data_loader, criterion):
    right_count = 0
    test_loss = 0.0
    for data in data_loader:
        inputs, labels = data
        p_outputs = net(inputs)  # forward
        right_count += torch.eq(torch.max(p_outputs.data, 1).indices, labels).sum().item()
        test_loss += criterion(p_outputs, labels).item()  # 累加loss
    return test_loss, right_count


# 绘制正确率图
def draw_right_rate_compare(test_right_rates, develop_right_rates):
    plt.figure(2)
    plt.plot(test_right_rates, label='train set')
    plt.plot(develop_right_rates, label='develop set', color='red', linestyle='--')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('right rate')
    plt.show()


# 绘制loss图
def draw_loss_compare(train_losses, develop_losses):
    plt.figure(1)
    plt.plot(train_losses, label='train set')
    plt.plot(develop_losses, label='develop set', color='red', linestyle='--')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()