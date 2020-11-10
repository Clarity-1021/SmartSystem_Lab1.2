from __future__ import print_function
import torch

x = torch.Tensor(5, 3)  # 构造一个未初始化的5*3的矩阵
x = torch.rand(5, 3)  # 构造一个随机初始化的矩阵
print('x=', x)  # 此处在notebook中输出x的值来查看具体的x内容
# x.size()
print('x_size=', x.size())
# NOTE: torch.Size 事实上是一个tuple, 所以其支持相关的操作*
y = torch.rand(5, 3)
print('y=', y)
# 此处将两个同形矩阵相加有两种语法结构
print('x + y=', x + y)  # 语法一
print('torch.add(x, y)=', torch.add(x, y))  # 语法二
# 另外输出tensor也有两种写法
# 语法一
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print('result=', result)
# 语法二
y.add_(x)  # 将y与x相加
print('y.add_(x)=', y)
# 特别注明：任何可以改变tensor内容的操作都会在方法名后加一个下划线'_'
# 例如：x.copy_(y), x.t_(), 这俩都会改变x的值。
# 另外python中的切片操作也是资次的。
print('x[:, 1]=', x[:, 1])  # 这一操作会输出x矩阵的第二列的所有值，以行向量的形式

# 此处演示tensor和numpy数据结构的相互转换
a = torch.ones(5)
print('a=', a)
b = a.numpy()
print('b=', b)

# 此处演示当修改numpy数组之后,与之相关联的tensor也会相应的被修改
a.add_(1)
print('a.add_(1)=', a)
print('b=', b)
# 将numpy的Array转换为torch的Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print('a=', a)
print('b=', b)

# 另外除了CharTensor之外，所有的tensor都可以在CPU运算和GPU预算之间相互转换
# 使用CUDA函数来将Tensor移动到GPU上
# 当CUDA可用时会进行GPU的运算
# if torch.cuda.is_available():
#     x = x.cuda()
#     print('x=', x)
#     y = y.cuda()
#     print('y=', y)
#     print('x + y=', x + y)

from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print('x=', x)
y = x + 2
print('y=', y)
y.creator

# y 是作为一个操作的结果创建的因此y有一个creator
z = y * y * 3
out = z.mean()

# 现在我们来使用反向传播
out.backward()

# out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
# 在此处输出 d(out)/dx
x.grad
