# mxnet学习之多层感知机(MLP)


<center >**前言**</center>

> 多层感知器算法，网上有各种解析，以后有时间自己在来慢慢补充。

<!--more-->

### 第二节课 mxnet之多层感知机

直接上代码：

```python 
# codin=utf-8

'''
从零开始的多层感知机算法实现
'''
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray as nd


def getModel():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(256, activation="relu"))
        net.add(gluon.nn.Dense(10))
    net.initialize()
    return net

# 数据格式转换


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32') / 255

# 加载数据


def readData():
    print("load data...")
    mnsit_train = gluon.data.vision.FashionMNIST(root="~/.mxnet/datasets/fashion-mnist",
                                                 train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(root="~/.mxnet/datasets/fashion-mnist",
                                                train=False	, transform=transform)
    return mnsit_train, mnist_test

# 画出图像


def drawPlot(images):
    import matplotlib.pyplot as plt
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()


def trainData(net, train, test, batch_size=128, epoches=5):
    train_data = gluon.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
                            "learning_rate": 0.5})
    for epoech in range(epoches):
        train_loss = 0
        train_acc = 0
        test_acc = 0
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
        print("Epoech is {0},train_loss is {1}.".format(
            epoech + 1, train_loss / len(train_data)))

# 准确率计算


def accuarcy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()

# 评价函数


def evaluateData(net, test, batch_size=128):
    acc = 0
    test_data = gluon.data.DataLoader(
        test, batch_size=batch_size, shuffle=False)
    for data, label in test_data:
        output = net(data)
        acc += accuarcy(output.astype("float32"), label.astype("float32"))
    acc = acc / len(test_data)
    print("test acc is {0}".format(acc))


train, test = readData()
net = getModel()
batch_size = 256
epoches = 5
trainData(net, train, test, batch_size, epoches)
evaluateData(net, test)
```

最后来个结果：多分类，这个结果真是惨不忍睹

![](http://owzdb6ojd.bkt.clouddn.com/17-12-5/39813172.jpg)
