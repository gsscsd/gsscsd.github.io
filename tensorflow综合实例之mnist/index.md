# tensorflow综合实例之MNIST


> 在本篇文章中，使用两种方法来做MNIST分类，一个是全连接层，一个是CNN。

> MNIST 数据集来自美国国家标准与技术研究所， National Institute of Standards and Technology (NIST)。训练集 (training set) 由来自 250 个不同人手写的数字构成，其中 50% 是高中学生，50% 来自人口普查局 (the Census Bureau) 的工作人员.。测试集(test set) 也是同样比例的手写数字数据。
>
> MNIST 数据集中有 60000 个训练样本和 10000 个测试样本，每个图片的大小是28×28。

<!--more -->

### MNIST分类0

> tensorflow实现MLP进行分类，使用两种方法来实现MLP，一种是手动前向计算，一种是使用tf.layersAPI。

```python
# 参考代码

# mnist分类实例
import tensorflow as tf 
import numpy as np 
from tqdm import tqdm

import matplotlib.pyplot as plt
# style设置
from matplotlib import style
from IPython import display
style.use('ggplot')
np.random.seed(42)
%matplotlib inline

data = np.load("./data/mnist.npz")
x_train,y_train,x_test,y_test = data['x_train'],data['y_train'],data["x_test"],data['y_test']

# 数据规整
x_train = x_train.reshape(-1,784) / 255 - 0.5
x_test = x_test.reshape(-1,784) / 255 - 0.5
# 这个能生成一个OneHot的10维向量，作为Y_train的一行，这样Y_train就有60000行OneHot作为输出
y_train = (np.arange(10) == y_train[:, None]).astype(int)  # 整理输出
y_test = (np.arange(10) == y_test[:, None]).astype(int) 

batch_size = 8 # 使用MBGD算法，设定batch_size为8

# batchsize的获取
def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys # 生成每一个batch
def modle_mlp_0(x):
    print("mlp0")
    # 第一隐藏层
    w1 = tf.Variable(tf.random_normal([784,400]))
    b1 = tf.Variable(tf.random_normal([400]))
    l1 = tf.nn.relu(tf.matmul(x,w1) + b1)
    print(l1)
    # 输出层
    w2 = tf.Variable(tf.random_normal([400,10]))
    b2 = tf.Variable(tf.random_normal([10]))
    out = tf.nn.softmax(tf.matmul(l1,w2) + b2)
    print(out)
    return out 

def modle_mlp_1(x):
    print("mlp1")
    # 使用dense层
    l1 = tf.layers.dense(x,400,tf.nn.relu)
    print(l1)
    # 输出层
    out = tf.layers.dense(l1,10,tf.nn.softmax)
    print(out)
    return out
tf.reset_default_graph()
# 输入层
tf_X = tf.placeholder(tf.float32,[None,784])
tf_Y = tf.placeholder(tf.float32,[None,10])
# model的输出
# mlp0
# out = modle_mlp_0(tf_X)
# mlp1 
out = model_mlp_1(tf_X)
# loss
loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(out,1e-11,1.0)))
# 优化
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 计算准确率
y_pred = tf.arg_max(out,1)
bool_pred = tf.equal(tf.arg_max(tf_Y,1),y_pred)
accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32)) # 准确率

num_epochs = 100 
accs = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs): # 迭代1000个周期
        for batch_xs,batch_ys in generatebatch(x_train,y_train,y_train.shape[0],batch_size): # 每个周期进行MBGD算法
            sess.run(train_step,feed_dict={tf_X:batch_xs,tf_Y:batch_ys})
        if(epoch % 10 == 0):
            res = sess.run(accuracy,feed_dict={tf_X:x_test,tf_Y:y_test})
            accs.append(res)
            print(epoch,res)
```

mlp0的acc输出结果：

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/mlp_mnist_0.png)

以下是mlp1的acc输出结果：

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/mlp_mnist_1.png)

### MNIST分类1

> tensorflow实现CNN进行分类，采用两种方式，一种是自定义权重和偏置前向计算，一种是使用默认的API。

```python
# mnist CNN分类实例
import tensorflow as tf 
import numpy as np 
from tqdm import tqdm

import matplotlib.pyplot as plt
# style设置
from matplotlib import style
from IPython import display
style.use('ggplot')
np.random.seed(42)
%matplotlib inline

data = np.load("./data/mnist.npz")
x_train,y_train,x_test,y_test = data['x_train'],data['y_train'],data["x_test"],data['y_test']

# 数据规整
x_train = x_train.reshape(-1,28,28,1) / 255 - 0.5
x_test = x_test.reshape(-1,28,28,1) / 255 - 0.5
# 这个能生成一个OneHot的10维向量，作为Y_train的一行，这样Y_train就有60000行OneHot作为输出
y_train = (np.arange(10) == y_train[:, None]).astype(int)  # 整理输出
y_test = (np.arange(10) == y_test[:, None]).astype(int) 

batch_size = 8 # 使用MBGD算法，设定batch_size为8

# batchsize的获取
def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys # 生成每一个batch

def model_cnn_0(tf_X):
    print("cnn1")
    # 第一卷积层+激活层
    conv_filter_w1 = tf.Variable(tf.random_normal([3, 3, 1, 10]))
    conv_filter_b1 =  tf.Variable(tf.random_normal([10]))
    relu_feature_maps1 = tf.nn.relu(tf.nn.conv2d(tf_X, conv_filter_w1,strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b1)
    print(relu_feature_maps1)
    # 池化层
    max_pool1 = tf.nn.max_pool(relu_feature_maps1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    print(max_pool1)
    # 第二卷积层
    conv_filter_w2 = tf.Variable(tf.random_normal([3, 3, 10, 5]))
    conv_filter_b2 =  tf.Variable(tf.random_normal([5]))
    conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2,strides=[1, 2, 2, 1], padding='SAME') + conv_filter_b2
    # BN归一化层+激活层 
    batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros([5]))
    scale = tf.Variable(tf.ones([5]))
    epsilon = 1e-3
    BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)
    relu_BN_maps2 = tf.nn.relu(BN_out)
    # 池化层
    max_pool2 = tf.nn.max_pool(relu_BN_maps2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    print(max_pool2)
    # 将特征图进行展开
    max_pool2_flat = tf.reshape(max_pool2, [-1, 7*7*5])
    # 全连接层
    fc_w1 = tf.Variable(tf.random_normal([7*7*5,50]))
    fc_b1 =  tf.Variable(tf.random_normal([50]))
    fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat, fc_w1) + fc_b1)
    # 输出层
    out_w1 = tf.Variable(tf.random_normal([50,10]))
    out_b1 = tf.Variable(tf.random_normal([10]))
    pred = tf.nn.softmax(tf.matmul(fc_out1,out_w1)+out_b1)
    return pred
  
def model_cnn_1(x):
    print("cnn2")
    #conv2 
    #layers.conv2d parameters
    #inputs 输入，是一个张量
    #filters 卷积核个数，也就是卷积层的厚度
    #kernel_size 卷积核的尺寸
    #strides: 扫描步长
    #padding: 边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
    #activation: 激活函数
    cn1 = tf.layers.conv2d(inputs=x,filters=10,kernel_size=(3,3),strides=1,padding="same",activation=tf.nn.relu)
    print(cn1)
    #tf.layers.max_pooling2d
    #inputs 输入，张量必须要有四个维度
    #pool_size: 过滤器的尺寸
    pool1 = tf.layers.max_pooling2d(inputs=cn1,pool_size=(3,3),strides=2,padding="same")
    print(pool1)
    #conv2 
    cn2 = tf.layers.conv2d(inputs=pool1,filters=5,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu)
    print(cn2)
    # bn归一化层
    batch_mean, batch_var = tf.nn.moments(cn2, [0, 1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros([5]))
    scale = tf.Variable(tf.ones([5]))
    epsilon = 1e-3
    BN_out = tf.nn.batch_normalization(cn2, batch_mean, batch_var, shift, scale, epsilon)
    relu_BN_maps2 = tf.nn.relu(BN_out)
    
    #pool2 2*2
    pool2 = tf.layers.max_pooling2d(inputs=relu_BN_maps2,pool_size=3,strides=2,padding="same")
    print(pool2)
    #flat(平坦化)
    flat = tf.reshape(pool2,[-1,7*7*5])
    dense = tf.layers.dense(inputs=flat,units=50,activation=tf.nn.relu)
    #输出层，不用激活函数（本质就是一个全连接层）
    out = tf.layers.dense(inputs = dense,units=10,activation=tf.nn.softmax)
    return out
  
tf.reset_default_graph()
# 输入层
tf_X = tf.placeholder(tf.float32,[None,28,28,1])
tf_Y = tf.placeholder(tf.float32,[None,10])
# model的输出
# 第一种cnn模型
out = model_cnn_0(tf_X)
# 第二种cnn模型
# out = model_cnn_1(tf_X)
# loss
loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(out,1e-11,1.0)))
# 优化
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 计算准确率
y_pred = tf.arg_max(out,1)
bool_pred = tf.equal(tf.arg_max(tf_Y,1),y_pred)
accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32)) # 准确率
num_epochs = 100 
accs = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs): # 迭代1000个周期
        for batch_xs,batch_ys in generatebatch(x_train,y_train,y_train.shape[0],batch_size): # 每个周期进行MBGD算法
            sess.run(train_step,feed_dict={tf_X:batch_xs,tf_Y:batch_ys})
        if(epoch % 10 == 0):
            res = sess.run(accuracy,feed_dict={tf_X:x_test,tf_Y:y_test})
            accs.append(res)
            print(epoch,res)
```

CNN0的acc输出结果：

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/cnn_mnist_0.png)

CNN1的acc输出结果：

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/cnn_mnist_1.png)


