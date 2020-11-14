from src import load_data
from src import tools
import torch
import torch.nn as nn

lab_index = '2'
epoch_index = '4'
batch_index = '6000'
rr = '99.167'
my_criterion = nn.CrossEntropyLoss()

net_save_path = '../lab_result/' + lab_index + '/jxw_cnn_net_' + epoch_index + '_' + batch_index + '_' + rr + "%.pkl"  # 训练好的卷积网络
params_save_path = '../lab_result/' + lab_index + '/jxw_cnn_net_params'  + epoch_index + '_' + batch_index + '_' + rr + "%.pkl"  # 训练好的卷积网络的参数


print('---Test_Start---')

# 加载网络和测试集
trained_net = torch.load(net_save_path)  # load net
test_loader, test_sample_size, test_batch_size = load_data.load_test_set()  # load lab_result set

# 测试用当前网络预测测试集中的数据的loss和正确个数
test_loss, test_right_count = tools.test_loss_and_right_count(trained_net, test_loader, my_criterion)
print('---Test_Finished---')

# 计算测试集的loss和正确率
test_loss = test_loss / len(test_loader)
test_right_rate = test_right_count / test_sample_size

# 打印结果
print('loss=%.5f,\t right rate=%.3f%%' % (test_loss, test_right_rate * 100))

