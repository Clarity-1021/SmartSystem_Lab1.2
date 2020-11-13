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

开发集和训练集都是`train`里面的图片，一大部分作为训练集（`600x12`），一小部分(`20x12`)划为开发集用作检查网络的准确性。

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

### 2.1 损失函数

对比了三种常用的损失函数：`MSELoss`、`L1Loss`、`CrossEntropyLoss`。

<center>
    <img src="./imgs/不同损失函数的对比实验/loss.png" width = "48%"/>
    <img src = "./imgs/不同损失函数的对比实验/right rate.png"  width = "48%"/>
</center>
<center><strong>图 1 - 不同损失函数在训练集上的效果对比</strong></center>

可以看出，交叉熵的效果最好，loss收敛的速度最快，分类正确率也远高于另外两个。

#### 结论

`CrossEntropyLoss`明显优于其他。

#### 论证

`MSELoss`和`L1Loss`都是以`one-hot`的输出来计算损失的，而`CrossEntropyLoss`是的每一个分类的输出是该分类的概率，精度远远高于前两者。
$$
Error=-\frac{1}{n}\sum_{j=1}^n{\sum_{i=1}^{k}{\left[ \left. G_{i}^{j}\ln \left( P_{i}^{j} \right) +\left( 1-G_{i}^{j} \right) \ln \left( 1-P_{i}^{j} \right) \right] \right.}}+\frac{\lambda}{2n}\sum_w{||w||^2}
$$

$$
\lambda :\text{正则项系数，}n:\text{样本个数，}k:\text{分类的种数（这里是12）}
$$

$$
G_{i}^{j}:\text{第}j\text{个样本第}i\text{种分类的理想概率, }P_{i}^{j}:\text{第}j\text{个样本第}i\text{种分类的实际概率}
$$

<center><strong>公式 1 - 交叉熵和正则项结合的损失函数</strong></center>

这个是我上次bp网的时候写的带正则项的交叉熵损失函数，可以看出来，它可以让正确的分类的概率的最大化，精度更高。

### 2.2 优化函数

对比了八种常用的优化函数：`SGD`、`SGD+Momentum`、`SGD+Momentum+Nesterov`、`Adagrad`、`Adadelta`、`RMSprop`、`Adam`、`Adamax`

其中`SGD+Momentum`、`SGD+Momentum+Nesterov`是`SGD`的改善版，

`Adadelta`、`RMSprop`、`Adam`是`Adagrad`的改善版，

`Adamax`是`Adam`的改善版。

`Adamax`比`Adam`增加了一个学习上限的概念。

<center>
    <img src="./imgs/不同优化函数的对比实验/train loss.png" width = "48%"/>
    <img src="./imgs/不同优化函数的对比实验/train right rate.png" width = "48%"/>
    <img src = "./imgs/不同优化函数的对比实验/develop loss.png"  width = "48%"/>
    <img src = "./imgs/不同优化函数的对比实验/develop right rate.png"  width = "48%"/>
</center>
<center><strong>图 2 - 不同优化函数效果对比(上-训练集、下-开发集)</strong></center>

两个较为基础的优化函数`SGD`和`Adagrad`效果显著差于其他更加完善的版本。

虽然`Adam`在训练集上的表现很好，但是在开发集上，经过多次迭代之后，loss竟然有明显的上升的趋势。

#### 结论

在开发集上，对于正确率和loss都表现最好的是`Adamax`。

#### 论证

##### `SGD`<`SGD+Momentum`<`SGD+Momentum+Nesterov`

`SGD`（随机梯度下降），会存在随机值的误差干扰，高方差时产生正当，会使得网络难以收敛。于是提出了`Momentum`（动量）的概念，随着梯度下降的时候，如果下降到一个较为平滑的区域，让下降的速度更慢一些，以免在最小值点附近来回震荡或者错过最小值点。但`Momentum`也会产生一个问题，会很容易让网络陷入一些鞍点或局部最小值。于是又增加了`Nesterov`的概念，`Nesterov+Momentum`是先根据`Momentum`，先根据上一次产生的动量移动之后再计算新的梯度，是对于`Momentum`的改善版本。但是对于我的网络来看，两者差别并不明显，同样的`Nesterov+Momentum`也有和`Momentum`一样的问题，容易陷入局部最小值和鞍点。

##### `Adagrad`<`Adadelta`、`RMSprop`<`Adam`<`Adamax`

`SGD`对于所有的参数都保持相同的学习率，`AdaGrad`（逐参数适应学习率法）对于不同的参数有不同的学习率。
$$
\mu =\sum_i{grad\left( \theta _i \right) ^2}
$$
$$
\theta _i:\text{第}i\text{个参数; }grad:\text{某参数当前迭代的梯度}
$$

<center><strong>公式 2 - 每一个参数的当前迭代的梯度平方和</strong></center>

每一次迭代会产生一个$\mu$，这是一个累计量，迭代次数越多越大，用全局学习率数除以这个累计量得到当前迭代每一个参数个性化的学习率，是参数自适应的，每个参数的学习率的大小受梯度和迭代次数两方面的影响，可以看出，迭代次数越多，以及该参数的梯度越大，学习率就会越小。带来的效果是前期的收敛是很好的，后期会变成惩罚收敛，速度越来越慢。`Adagrad`会累加之前所有的梯度平方，而`Adadelta`只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值。不依赖全局学习率，缺点是后期会在局部最小值附加抖动。`RMSprop`用全局学习率除以均方根，减缓学习率的下降。 `Adadelta`和`RMSprop`都是对于`Adagrad`使得学习率下降过快的不同的改善版本。`Adam`可以看做是带`Momentum`的`RMSprop`，比`RMSprop`的效果更好。`Adamax`是`Adam`的一个变种，增加了一个学习率上限的概念，把学习率的边界范围更加简单。

而从我的实验来看，似乎`Adagrad`及其衍生的优化函数的表现并不比`SGD`及其衍生好很多，但`Adamax`的效果优于`SGD+Momentum+Nesterov`的版本，所以最后选择前者。

### 2.3 batch_size 和 learning rate

<center>
    <img src = "./imgs/AdamMax_batchsize=5/develop loss.png"  width = "48%"/>
    <img src="./imgs/AdamMax_batchsize=5/develop right rate.png" width = "48%"/>
</center>
<center><strong>图 3 - batchsize=5时开发集相关数据</strong></center>

<center>
    <img src = "./imgs/AdamMax_batchsize=2/develop loss.png"  width = "48%"/>
    <img src="./imgs/AdamMax_batchsize=2/develop right rate.png" width = "48%"/>
</center>
<center><strong>图 4 - batchsize=2时开发集相关数据</strong></center>

<center>
    <img src = "./imgs/AdamMax_batchsize=1/develop loss.png"  width = "48%"/>
    <img src="./imgs/AdamMax_batchsize=1/develop right rate.png" width = "48%"/>
</center>
<center><strong>图 5 - batchsize=1时开发集相关数据</strong></center>

#### 结论

batch_size=1，learning_rate=0.01时最佳。在第6个epoch和第10个epoch时峰值达到过**99.167%**。

#### 论证

全局学习率大一点，每一个样本就调一次效果最好的原因，可能是调整的次数增加，对每一个数据的适应力就更好了。

### 2.4 标准化

因为我本来就是标准化处理过的，这个对比一下去掉标准化的情况。由于我的网络是有两个隐藏层的，我对比只对第一层标准化，只对第二次标准化和完全没有不标准化的网络各是什么效果。

<center>
    <img src = "./imgs/正则化对比实验/train loss.png"  width = "48%"/>
    <img src="./imgs/正则化对比实验/train right rate.png" width = "48%"/>
    <img src = "./imgs/正则化对比实验/develop loss.png"  width = "48%"/>
    <img src="./imgs/正则化对比实验/develop right rate.png" width = "48%"/>
</center>
<center><strong>图 6 - 正则化对比实验(上-训练集、下-开发集)</strong></center>

可以看到没有完全经过标准化的红线，不仅在最开始的几次迭代中，显著的表现很差之外，在开发集上loss过早的从底部反弹，趋势不再收敛。

而只有一层隐藏层标准化的网络，只有第一层标准化的网络表现比只有最后一层隐藏层正则化的网络效果要来的差。

#### 结论

每一层隐藏层都标准化，网络效果最好。

#### 论证



### 2.5 图片处理

由于原训练的图片像素都是`28x28`，我这里尝试对图片进行像素重置，提高到`32x32`进行对比。

<center>
    <img src = "./imgs/提高图片分辨率对比实验/train loss.png"  width = "48%"/>
    <img src="./imgs/提高图片分辨率对比实验/train right rate.png" width = "48%"/>
    <img src = "./imgs/提高图片分辨率对比实验/develop loss.png"  width = "48%"/>
    <img src="./imgs/提高图片分辨率对比实验/develop right rate.png" width = "48%"/>
</center>
<center><strong>图 7 - 重置提高图片分辨率对比实验(上-训练集、下-开发集)</strong></center>

**99.167%**是目前网络在开发集上能达到的正确率峰值。可以看到两个都可以达到这个峰值，并且两者在训练集上的表现非常相似。但是在开发集的loss曲线上，`28x28`的网络随着跌倒次数的增加，loss的波动逐渐显著，相比之下`32x32`更加平滑（虽然也没有很平滑）。

#### 结论

`32x32`的网络在开发集上loss的收敛表现更好。

### 2.6 网络结构

原本的网络是有两个隐藏层，第一个隐藏层的神经元个数的6个（蓝线），第二层是16个。显著尝试改变隐藏藏层的层数，没每一层的神经元个数。

<center>
    <img src = "./imgs/网络层数和神经元个数不同对比实验/train loss.png"  width = "48%"/>
    <img src="./imgs/网络层数和神经元个数不同对比实验/train right rate.png" width = "48%"/>
    <img src = "./imgs/网络层数和神经元个数不同对比实验/develop loss.png"  width = "48%"/>
    <img src="./imgs/网络层数和神经元个数不同对比实验/develop right rate.png" width = "48%"/>
</center>
<center><strong>图 8 - 网络层数和神经元个数不同对比实验(上-训练集、下-开发集) (1)</strong></center>

<center>
    <img src = "./imgs/网络层数-2/train loss.png"  width = "48%"/>
    <img src="./imgs/网络层数-2/train right rate.png" width = "48%"/>
    <img src = "./imgs/网络层数-2/develop loss.png"  width = "48%"/>
    <img src="./imgs/网络层数-2/develop right rate.png" width = "48%"/>
</center>
<center><strong>图 9 - 网络层数和神经元个数不同对比实验(上-训练集、下-开发集) (2)</strong></center>

由于第一次的，各卷积神经网络的表现差异并不明显，就做了第二次实验。但是可以看出只有一层隐藏层的网络在开发集上显著的表现差于其他。但网络层数也不是越多越好，可以看到第一、二次实验中，红线的网络随迭代次数增加，正确率的波动明显增大。而紫线和蓝线的表现并没有过大的区别，所以也不需要增加武威的第三层。

#### 结论

两层的神经网络在各层输出个数合理的情况下，就很够用了。

## 3 对网络设计的理解

卷积神经网络，简单来说就是要顺序执行卷积，激活，池化，最后将最后一层隐藏层提取的特征，线性变换得到最后的分类输出。

隐藏层的层数，各层的神经元个数，卷积的窗口大小，移动的步长，池化的窗口大小，移动的步长，以及是否进行正则化都会对网络的实际效果产生影响。

要想设计出好的网络，网络的学习率，batch_size，以及以上的各个因素都要综合考虑。




