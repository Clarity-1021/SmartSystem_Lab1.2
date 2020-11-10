## 多线程错误

https://blog.csdn.net/u013700358/article/details/82753019

# ![Pytorch出错](https://img-blog.csdn.net/20180918095938413?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MDAzNTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

其实挺奇怪的… 因为这段代码是前几天刚写完跑成功的，今天再跑就出问题了
至于解决方法也很简单，出错提到了是多线程的缘故，那么就有如下两种：

1. 去掉`num_workers`参数

```python
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
1212
```

1. 在跑`epoch`之前，加上`if __name__=='__main__'`:

```python
if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
123123
```

这样就能正常运行了
至于`if __name__=='__main__`的作用

> 当.py文件被**直接运行**时，`if __name__ == '__main__'`之下的代码块将被运行；
> 当.py文件**以模块形式被导入**时，`if __name__ == '__main__'`之下的代码块不被运行。

可参考[Python 中的 if **name** == ‘**main**’ 该如何理解](http://blog.konghy.cn/2017/04/24/python-entry-program/)



## `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`

 代码里加两行

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

## 用自己图片训练/测试/保存

https://blog.csdn.net/weixin_41770169/article/details/90750965

`yyds`

