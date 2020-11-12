# `SmartSystem_Lab1.2`

##### 18307130251 蒋晓雯

###### 有能力的同学可以进一步查询相关资料，使用一些方法改进网络，提高精度。

###### 编写实验文档，可以包括但不限于:

- ###### 代码基本结构

- ###### 设计实验改进网络并论证

- ###### 对网络设计的理解

###### 提交: 将所有代码、文档放在 学号-姓名 文件夹下，打包上传到`WORK_UPLOAD/LAB1/PART2` 目录下。

## 1 代码基本结构

### 卷积网络 `src/CNN_network.py`

设置了两个隐藏层。

### 加载数据 `src/load_data.py`

```python
torchvision.datasets.ImageFolder
```

使用`torchvision`加载训练集和开发集的数据。先将图片转化为灰度图再转为张量。

训练集进行混洗，开发集不混洗。

开发集和训练集都是`train`里面的图片，一大部分作为训练集，一小部分划为开发集用作检查网络的准确性。

### 训练网络 `src/main.py`

1. 加载训练集和开发集的数据，设置损失函数和优化器

2. 每一个epoch中对于训练集中的数据，分批进行：

   - 梯度清零
   - 正向传播，用网络当前的权重得到预测结果
   - 用机器预测结果和理想分类结果过损失函数，得到loss
   - 通过loss对网络进行反向传播，调整网络的权重
   - 用优化函数根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
   - 累加当前批次的loss

3. 一个epoch结束之后，用当前网络的权重测试开发集中图片分类的正确率和loss，输出；同时也输出训练集当前epoch的loss和分类正确率

4. 存每个epoch训练集和开发集的loss和正确率

5. 训练结束之后，保存网络和网络参数

6. 用上面存的loss和正确率数据画图

   ###### *7.可以重新加载网络来测试别的测试集，由于没有新的数据，我这里是测试的还是我的开发集

## 2 设计实验改进网络并论证

优化技巧

https://www.cnblogs.com/wj-1314/p/9838866.html

`batchnormalization`

https://blog.csdn.net/bigFatCat_Tom/article/details/91619977

### 2.1 损失函数

![loss](D:\Wendy\学习\人工智能\Labs\Lab1.2\SmartSystem_Lab1.2\imgs\不同损失函数的对比实验\loss.png)

![right rate](D:\Wendy\学习\人工智能\Labs\Lab1.2\SmartSystem_Lab1.2\imgs\不同损失函数的对比实验\right rate.png)

#### 



### 2.2 优化函数

7种优化函数

https://blog.csdn.net/qq_36589234/article/details/89330342

10种优化函数

https://blog.csdn.net/u011995719/article/details/88988420?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.edu_weight&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.edu_weight

#### 2.2.1 `SGD(stochastic gradient descent)`

```python
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

参数：

- params (iterable)：iterable of parameters to optimize or dicts defining parameter groups
- lr (float)：learning rate
- momentum (float, optional)：momentum factor (default: 0)
- weight_decay (float, optional)：weight decay (L2 penalty) (default: 0)即L2regularization，选择一个合适的权重衰减系数λ非常重要，这个需要根据具体的情况去尝试，初步尝试可以使用 `1e-4` 或者 `1e-3`
- dampening (float, optional)：dampening for momentum (default: 0)
- nesterov (bool, optional)：enables Nesterov momentum (default: False)

##### 2.2.1.1 `Momentum`

通过交叉验证，这个参数通常设为[0.5,0.9,0.95,0.99]中的一个，一般为0.9。

##### 2.2.1.2 `Nesterov Momentum`

设置momentum同时设置`nesterov`

```python
nesterov=True
```

#### 2.2.2 `AdaGrad(Pre-parameter adaptive learning rate methods)`

```python
torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
```

##### 参数：

- params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
- lr (float, optional) – learning rate (default: 1e-2)
- lr_decay (float, optional) – learning rate decay (default: 0)
- weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)

#### 2.2.3 `RMSProp(AdaGrad的梯度平方滑动平均版)`

```python
torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

参数：

- params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
- lr (float, optional) – learning rate (default: 1e-2)
- momentum (float, optional) – momentum factor (default: 0)
- alpha (float, optional) – smoothing constant (default: 0.99)
- eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)
- centered (bool, optional) – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
- weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)

#### 2.2.4 `Adam(RMSProp的Momentum版)`

```python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

参数：

- params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
- lr (float, optional) – learning rate (default: 1e-3)
- betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
- eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)
- weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
- amsgrad (boolean, optional) – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: False)

## 3 对网络设计的理解