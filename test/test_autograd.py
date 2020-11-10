import torch
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print('x=', x)
y = x + 2
print('y=', y)
y.creator

# y 是作为一个操作的结果创建的因此y有一个creator
z = y * y * 3
out = z.mean()
