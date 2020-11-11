import CNN_network
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


my_transforms = transforms.Compose([transforms.Grayscale(),  # 转灰度图
                                    transforms.ToTensor()])  # 转为Tensor
# train_size = "600"  # 每个文字分类取的训练集的个数
train_size = "500"

# train_set_prams
train_set_path = r"./data/" + train_size + "/train"  # 训练集样本的文件夹
train_batch_size = 5  # 训练集的batch_size
# develop_set_prams
develop_set_path = r"./data/" + train_size + "/develop"  # 开发集样本的文件夹
develop_batch_size = 20  # 开发集的batch_size
# test_set_prams
test_set_path = develop_set_path  # 测试集样本的文件夹
test_batch_size = 20  # 测试集的batch_size

# network_save
net_save_path = 'jxw_cnn_net.pkl'  # 训练好的卷积网络存储的地址
params_save_path = 'jxw_cnn_net_params.pkl'  # 训练好的卷积网络的参数存储的地址

running_check_loss_mode = False  # True:每一个epoch显式好次loss和正确率, False:每一个epoch显式一次loss和正确率
running_sample_size = 200

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


# 分类名
classes = ('博', '学', '笃', '志', '切', '问', '近', '思', '自', '由', '无', '用')


# train
def train_save_net():
    net = CNN_network.CNN_network()
    optimizer = my_optimizer(net)
    criterion = my_criterion
    train_loader, train_sample_size = load_train_set()
    develop_loader, develop_sample_size = load_develop_set()

    print('---Data_got---')
    print('---CNN_NET_Train_Start---')

    if __name__ == '__main__':
        for epoch in range(epoch_count):
            print('epoch[%2d/%2d]' % (epoch + 1, epoch_count))
            train_loss = 0.0
            train_right_count = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                p_outputs = net(inputs)  # forward
                loss = criterion(p_outputs, labels)
                loss.backward()  # backward
                optimizer.step()  # optimizer
                train_loss += loss.item()
                train_right_count = update_right_count_by_batch(train_right_count, labels, p_outputs)

                if running_check_loss_mode:
                    if i % running_sample_size == 199:
                        develop_loss, develop_right_count = test_loss_and_right_rate(net, develop_loader, criterion)
                        print('[%4d/%4d]' % ((i + 1), train_sample_size))
                        print('running train loss=%.5f,\t develop loss=%.5f' %
                              (train_loss / running_sample_size, develop_loss / develop_sample_size))
                        print('train right rate=%.3f%%,\t develop right rate=%.3f%%' %
                              (((train_right_count / train_sample_size) * 100),
                               ((develop_right_count / develop_sample_size) * 100)))
                        train_loss = 0.0

            if not running_check_loss_mode:
                develop_loss, develop_right_count = test_loss_and_right_rate(net, develop_loader, criterion)
                print('train loss=%.5f,\t train right rate=%.3f%%' %
                      (train_loss / train_sample_size,
                       (train_right_count / train_sample_size) * 100))
                print('develop loss=%.5f,\t develop right rate=%.3f%%' %
                      (develop_loss / develop_sample_size,
                       (develop_right_count / develop_sample_size) * 100))
    print('---CNN_NET_Train_Finished---')

    torch.save(net, net_save_path)
    torch.save(net.state_dict(), params_save_path)
    print('---CNN_NET_Saved---')


def update_right_count_by_batch(right_count, labels, p_outputs):
    p_outputs_size = p_outputs.size(0)
    predicts = torch.max(p_outputs.data, 1).indices
    # print(torch.max(p_outputs.data, 1))
    for output_index in range(p_outputs_size):
        if predicts[output_index] == labels[output_index]:
            right_count = right_count + 1
    return right_count


def test_loss_and_right_rate(net, data_loader, criterion):
    right_count = 0
    test_loss = 0.0
    for data in data_loader:
        # get the inputs
        inputs, labels = data
        p_outputs = net(inputs)
        right_count = update_right_count_by_batch(right_count, labels, p_outputs)
        # loss = criterion(p_outputs, labels)
        test_loss += criterion(p_outputs, labels).item()
        # test_loss += loss.item()
    return test_loss, right_count


# test
def test():
    trained_net = torch.load(net_save_path)  # load net
    test_loader, test_sample_size = load_test_set()  # load test data set
    test_loss, test_right_count = test_loss_and_right_rate(trained_net, test_loader,
                                                           my_criterion)
    print('---Test_Finished---')
    print('test loss=%.5f,\t test right rate=%.3f%%' %
          (test_loss / test_sample_size,
           (test_right_count / test_sample_size) * 100))


train_save_net()
test()