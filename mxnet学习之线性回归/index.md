# mxnet学习之线性回归


<center >**前言**</center>

> 深度学习模型太多了，tensorflow、theano、keras、mxnet等等，听说mxnet开发者们在斗鱼直播教学，所以趁机学习一波。

<!--more-->

### 第一节课 从零开始之线性回归

#### 先从零开始，搭一个线性回归

```python
# coding=utf-8

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt
import random


def data_iter(X, y, num_examples, batch_size):
    # 产生索引
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)


def net(X, w, b):
    return nd.dot(X, w) + b


def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def main():

    num_inputs = 2
    num_examples = 1000

    true_w = [2, -3.4]
    true_b = 4.2

    X = nd.random_normal(shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += 0.01 * nd.random_normal(shape=y.shape)

    batch_size = 10

    # # 画图
    # plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
    # plt.show()
    # for data, label in data_iter(X, y, num_examples, batch_size):
    #     print(data, label)
    #     break
    w = nd.random_normal(shape=(num_inputs, 1))
    b = nd.zeros((1,))
    params = [w, b]
    for param in params:
        param.attach_grad()

    epochs = 5
    learning_rate = 0.01

    for e in range(epochs):
        total_loss = 0
        for data, label in data_iter(X, y, num_examples, batch_size):
            with autograd.record():
                output = net(data, w, b)
                loss = square_loss(output, label)
            loss.backward()
            SGD(params, learning_rate)
            total_loss += nd.sum(loss).asscalar()
        print("Epoch %d,average loss is %f" % (e, total_loss / num_examples))
    print(true_w, w)
    print(true_b, b)
    pass
if __name__ == '__main__':
    main()
```

#### 使用gluon搭建线性回归

```python 
# coding=utf-8

from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon.nn import Dense


def main():
    num_inputs = 2
    num_examples = 1000

    true_w = [2, -3.4]
    true_b = 4.2

    X = nd.random_normal(shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += 0.01 * nd.random_normal(shape=y.shape)
    # print("x is ", X[:3])
    # print("y is ", y[:3])

    batch_size = 10
    dataset = gluon.data.ArrayDataset(X, y)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

    # for data, label in data_iter:
    #     print(data, label)
    #     break
    net = gluon.nn.Sequential()
    net.add(Dense(1))
    net.initialize()
    square_loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {
                            "learning_rate": 0.1})
    epochs = 5
    for e in range(epochs):
        total_loss = 0
        for data, label in data_iter:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            total_loss += nd.sum(loss).asscalar()
        print("Epoch %d,average loss is %f" % (e, total_loss / num_examples))
    dense = net[0]
    print(true_w, dense.weight.data())
    print(true_b, dense.bias.data())
    
if __name__ == '__main__':
    main()
```


