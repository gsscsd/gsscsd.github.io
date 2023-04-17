# keras基本入门




### keras基本介绍

> Keras是由纯python编写的基于不同的深度学习后端开发的深度学习框架。
>
> 支持的后端有：
>
> - 谷歌的 TensorFlow 后端
> - 微软的 CNTK 后端
> - Theano 后端
>
> Keras是一个高层神经网络API，支持快速实验，能够把你的idea迅速转换为结果，如果有如下需求，可以优先选择Keras：
>
> - 简易和快速的原型设计（keras具有高度模块化，极简，和可扩充特性）
>
> - 支持CNN和RNN，或二者的结合
>
> - 无缝CPU和GPU切换
>
> keras的优点：
>
> - 用户友好：Keras是为人类而不是天顶星人设计的API。用户的使用体验始终是我们考虑的首要和中心内容。Keras遵循减少认知困难的最佳实践：Keras提供一致而简洁的API， 能够极大减少一般应用下用户的工作量，同时，Keras提供清晰和具有实践意义的bug反馈。
>
> - 模块性：模型可理解为一个层的序列或数据的运算图，完全可配置的模块可以用最少的代价自由组合在一起。具体而言，网络层、损失函数、优化器、初始化策略、激活函数、正则化方法都是独立的模块，你可以使用它们来构建自己的模型。
>
> - 易扩展性：添加新模块超级容易，只需要仿照现有的模块编写新的类或函数即可。创建新模块的便利性使得Keras更适合于先进的研究工作。
>
> - 与Python协作：Keras没有单独的模型配置文件类型（作为对比，caffe有），模型由python代码描述，使其更紧凑和更易debug，并提供了扩展的便利性。

<!--more-->

### keras模块

#### keras的整体框架

![img](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/keras.png)

#### keras搭建神经网络的步骤

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/keras_step.png)

### keras的安装

```python
# GPU 版本
pip install --upgrade tensorflow-gpu

# CPU 版本
pip install --upgrade tensorflow

# Keras 安装
pip install keras -U --pre
```

### keras实例

#### keras入门实例之回归模型

> Keras中定义一个单层全连接网络，进行线性回归模型的训练

```python
import numpy as np
np.random.seed(42)  
from keras.models import Sequential 
from keras.layers import Dense
import matplotlib.pyplot as plt
%matplotlib inline

# 创建数据集,首先生成200个从-1,1的数据
X = np.linspace(-1, 1, 2000)
np.random.shuffle(X)    # 打乱数据
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (2000, )) # 假设我们真实模型为：Y=0.5X+2

X_train, Y_train = X[:1600], Y[:1600]     # 把前160个数据放到训练集
X_test, Y_test = X[1600:], Y[1600:]       # 把后40个点放到测试集

# 定义一个model，
model = Sequential () # Keras有两种类型的模型，序贯模型（Sequential）和函数式模型
                      # 比较常用的是Sequential，它是单输入单输出的
model.add(Dense(output_dim=1, input_dim=1)) # 通过add()方法一层层添加模型
                                            # Dense是全连接层，第一层需要定义输入，
                                            # 第二层无需指定输入，一般第二层把第一层的输出作为输入

# 定义完模型就需要训练了，不过训练之前我们需要指定一些训练参数
# 通过compile()方法选择损失函数和优化器
# 这里我们用均方误差作为损失函数，随机梯度下降作为优化方法
model.compile(loss='mse', optimizer='sgd')

# 开始训练
print('Training -----------')
for step in range(3001):
    cost = model.train_on_batch(X_train, Y_train) # Keras有很多开始训练的函数，这里用train_on_batch（）
    if step % 500 == 0:
        print('train cost: ', cost)

# 测试训练好的模型
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()    # 查看训练出的网络参数
                                        # 由于我们网络只有一层，且每次训练的输入只有一个，输出只有一个
                                        # 因此第一层训练出Y=WX+B这个模型，其中W,b为训练出的参数
```

输出的结果为：

```python
Training -----------
train cost:  4.6456146
train cost:  0.0033609855
train cost:  0.0024392079
train cost:  0.0024380628
train cost:  0.002438061
train cost:  0.002438061
train cost:  0.002438061

Testing ------------
400/400 [==============================] - 0s 916us/step
test cost: 0.0025163212092593314
```

```python
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred,color='red')
plt.show()

# 输出权重以及可视化结果
# Weights= [[0.5028866]] 
# biases= [2.0018716]
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/keras_mse.png)

从上图可以看出，结果还是相对比较准确的。

#### keras入门实例之手写数字识别

> 使用全连接层神经网络来预测手写数字

```python
from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.optimizers import SGD,RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# 参数设置
tBatchSize = 16 # 批处理的大小
tEpoches = 20  # 迭代次数


'''第一步：选择模型'''
model = Sequential() # 采用贯序模型
 
'''第二步：构建网络层'''
'''构建网络只是构建了一个网络结构，并定义网络的参数，此时还没有输入的数据集'''
#构建的第一个层作为输入层
# Dense 这是第一个隐藏层，并附带定义了输入层，该隐含层有400个神经元。输入则是 784个节点
model.add(Dense(400,input_shape=(784,))) # 输入层，28*28=784 输入层将二维矩阵换成了一维向量输入
model.add(Activation('relu')) # 激活函数是relu
#model.add(Dropout(0.5)) # 采用50%的dropout  随机取一半进行训练
 
 
#构建的第3个层作为输出层
model.add(Dense(10)) # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax')) # 最后一层用softmax作为激活函数
 
'''第三步：网络优化和编译'''
#   lr：大于0的浮点数，学习率
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
 
# 只有通过了编译，model才真正的建立起来，这时候才能够被使用
#model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical') # 使用交叉熵作为loss函数 这是原例子，但是执行出错
model.compile(loss='categorical_crossentropy', 
              optimizer=rmsprop,
              metrics=['accuracy']) 

print(model.summary())
# 模型的结果为：
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_16 (Dense)             (None, 400)               314000    
_________________________________________________________________
activation_16 (Activation)   (None, 400)               0         
_________________________________________________________________
dense_17 (Dense)             (None, 10)                4010      
_________________________________________________________________
activation_17 (Activation)   (None, 10)                0         
=================================================================
Total params: 318,010
Trainable params: 318,010
Non-trainable params: 0
_________________________________________________________________
"""
'''第四步：训练'''
# 数据集获取 mnist
data = np.load("./mnist/mnist.npz")
X_train,y_train,X_test,y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']

# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
 
#这个能生成一个OneHot的10维向量，作为Y_train的一行，这样Y_train就有60000行OneHot作为输出
Y_train = (np.arange(10) == y_train[:, None]).astype(int)  # 整理输出
Y_test = (np.arange(10) == y_test[:, None]).astype(int)    #np.arange(5) = array([0,1,2,3,4])
'''
   model.fit的一些参数
   batch_size：对总的样本数进行分组，每组包含的样本数量
   epochs ：训练次数
   shuffle：是否把数据随机打乱之后再进行训练
   validation_split：拿出百分之多少用来做交叉验证
   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
'''
model.fit(X_train, Y_train, 
          batch_size=tBatchSize, 
          epochs=tEpoches, 
          shuffle=True, 
          validation_split=0.3
          )
```

输出结果为：

```python
Train on 42000 samples, validate on 18000 samples
Epoch 1/20
42000/42000 [==============================] - 31s 728us/step - loss: 5.8465 - acc: 0.6261 - val_loss: 4.4380 - val_acc: 0.7159
Epoch 2/20
42000/42000 [==============================] - 27s 643us/step - loss: 4.2894 - acc: 0.7277 - val_loss: 4.0947 - val_acc: 0.7392
Epoch 3/20
42000/42000 [==============================] - 26s 608us/step - loss: 4.1076 - acc: 0.7397 - val_loss: 3.9447 - val_acc: 0.7493
Epoch 4/20
42000/42000 [==============================] - 24s 572us/step - loss: 3.9682 - acc: 0.7488 - val_loss: 3.9679 - val_acc: 0.7487
Epoch 5/20
42000/42000 [==============================] - 21s 508us/step - loss: 3.8599 - acc: 0.7556 - val_loss: 3.7947 - val_acc: 0.7604
Epoch 6/20
42000/42000 [==============================] - 16s 380us/step - loss: 3.8004 - acc: 0.7601 - val_loss: 3.7608 - val_acc: 0.7627
Epoch 7/20
42000/42000 [==============================] - 10s 234us/step - loss: 3.7333 - acc: 0.7647 - val_loss: 3.7849 - val_acc: 0.7608
Epoch 8/20
42000/42000 [==============================] - 8s 196us/step - loss: 3.6918 - acc: 0.7676 - val_loss: 3.6805 - val_acc: 0.7679
Epoch 9/20
42000/42000 [==============================] - 5s 130us/step - loss: 3.6581 - acc: 0.7700 - val_loss: 3.7473 - val_acc: 0.7636
Epoch 10/20
42000/42000 [==============================] - 4s 103us/step - loss: 3.6168 - acc: 0.7722 - val_loss: 3.6767 - val_acc: 0.7676
Epoch 11/20
42000/42000 [==============================] - 5s 112us/step - loss: 2.7719 - acc: 0.8233 - val_loss: 2.3355 - val_acc: 0.8493
Epoch 12/20
42000/42000 [==============================] - 4s 106us/step - loss: 2.2019 - acc: 0.8596 - val_loss: 2.2233 - val_acc: 0.8576
Epoch 13/20
42000/42000 [==============================] - 4s 99us/step - loss: 2.1230 - acc: 0.8652 - val_loss: 2.1143 - val_acc: 0.8650
Epoch 14/20
42000/42000 [==============================] - 4s 98us/step - loss: 2.0812 - acc: 0.8677 - val_loss: 2.1907 - val_acc: 0.8598
Epoch 15/20
42000/42000 [==============================] - 4s 100us/step - loss: 2.0510 - acc: 0.8697 - val_loss: 2.1616 - val_acc: 0.8611
Epoch 16/20
42000/42000 [==============================] - 4s 103us/step - loss: 2.0189 - acc: 0.8718 - val_loss: 2.1881 - val_acc: 0.8597
Epoch 17/20
42000/42000 [==============================] - 4s 100us/step - loss: 2.0193 - acc: 0.8718 - val_loss: 2.0779 - val_acc: 0.8669
Epoch 18/20
42000/42000 [==============================] - 4s 92us/step - loss: 1.0165 - acc: 0.9312 - val_loss: 0.8832 - val_acc: 0.9387
Epoch 19/20
42000/42000 [==============================] - 4s 94us/step - loss: 0.7048 - acc: 0.9512 - val_loss: 0.8085 - val_acc: 0.9443
Epoch 20/20
42000/42000 [==============================] - 4s 95us/step - loss: 0.6226 - acc: 0.9570 - val_loss: 0.8320 - val_acc: 0.9418
```

可视化训练过程的结果为：

```python
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.legend(['loss',"val_loss","acc","val_acc"])
plt.show()
```



![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/acc_loss_mnist.png)

```python
'''第五步：预测'''
print("test set")
# 误差评价 ：按batch计算在batch用到的输入数据上模型的误差
loss, accuracy = model.evaluate(X_test,Y_test,verbose=0)
print('test loss: ', loss)
print('test accuracy: ', accuracy)
#---------------------------
#test set
#test loss:  0.7981250540712492
#test accuracy:  0.9453

# 根据模型获取预测结果  为了节约计算内存，也是分组（batch）load到内存中的，
result = model.predict(X_test,batch_size=tBatchSize,verbose=1)
 
# 找到每行最大的序号
result_max = np.argmax(result, axis = 1) #axis=1表示按行 取最大值   如果axis=0表示按列 取最大值 axis=None表示全部
test_max = np.argmax(Y_test, axis = 1) # 这是结果的真实序号
 
result_bool = np.equal(result_max, test_max) # 预测结果和真实结果一致的为真（按元素比较）
true_num = np.sum(result_bool) #正确结果的数量
print("The accuracy of the model is %f" % (true_num/len(result_bool))) # 验证结果的准确率
#--------------------------
# 10000/10000 [==============================] - 1s 129us/step
# The accuracy of the model is 0.945300
```

#### keras入门实例之卷积神经网络

> 使用cnn来做手写数字识别

```python
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,MaxPool2D,Convolution2D
import numpy as np 
np.random.seed(42)
from keras.utils import np_utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
%matplotlib inline

# 数据集获取 mnist
data = np.load("./data/mnist.npz")
X_train,y_train,X_test,y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']
# 注意一下，由于后端是tf，所以维度应该是(width,height,channels)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

model = Sequential()
# 添加CNN层
# CNN层,filter的个数为32,filter的大小为(5,5),输出尺寸为(28,28.32)
model.add(Convolution2D(filters=32,kernel_size=5,input_shape=(28,28,1)))
model.add(Activation("relu"))
# pool层,采用maxpool,输出尺寸为(14,14,32)
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
# CNN层,filter的个数为64,filter的大小为(5,5),输出尺寸为(14,14,64)
model.add(Convolution2D(filters=64,kernel_size=(5,5),padding="same"))
model.add(Activation("relu"))
# pool层,采用maxpool,输出尺寸为(7,7,64)
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
# Flatten层,用来接入全连接层,输出尺寸为(64 * 7 * 7) 
model.add(Flatten())
# Dense层,输出尺寸为1024
model.add(Dense(1024))
model.add(Activation("relu"))
# Dense输出层.10分类
model.add(Dense(10))
model.add(Activation("softmax"))

adam = Adam(lr = 1e-4)
model.compile(loss = 'categorical_crossentropy',
              optimizer=adam,
             metrics=['accuracy'])
print(" Training ---------------")
hist = model.fit(X_train,y_train,batch_size=32,epochs=6)

"""
 Training ---------------
Epoch 1/6
60000/60000 [==============================] - 15s 249us/step - loss: 0.3424 - acc: 0.9084
Epoch 2/6
60000/60000 [==============================] - 14s 233us/step - loss: 0.1012 - acc: 0.9692
Epoch 3/6
60000/60000 [==============================] - 14s 233us/step - loss: 0.0640 - acc: 0.9794
Epoch 4/6
60000/60000 [==============================] - 14s 228us/step - loss: 0.0452 - acc: 0.9854
Epoch 5/6
60000/60000 [==============================] - 14s 226us/step - loss: 0.0344 - acc: 0.9892
Epoch 6/6
60000/60000 [==============================] - 13s 222us/step - loss: 0.0252 - acc: 0.9920
"""
print(model.summary())
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_14 (Conv2D)           (None, 24, 24, 32)        832       
_________________________________________________________________
activation_25 (Activation)   (None, 24, 24, 32)        0         
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 12, 12, 32)        0         
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 12, 12, 64)        51264     
_________________________________________________________________
activation_26 (Activation)   (None, 12, 12, 64)        0         
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 6, 6, 64)          0         
_________________________________________________________________
flatten_7 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_13 (Dense)             (None, 1024)              2360320   
_________________________________________________________________
activation_27 (Activation)   (None, 1024)              0         
_________________________________________________________________
dense_14 (Dense)             (None, 10)                10250     
_________________________________________________________________
activation_28 (Activation)   (None, 10)                0         
=================================================================
Total params: 2,422,666
Trainable params: 2,422,666
Non-trainable params: 0
_________________________________________________________________
"""
print("Test -------------")
loss,accuray = model.evaluate(X_test,y_test)
print("loss is :",loss)
print("scores is :",accuray)
# Test -------------
# 10000/10000 [==============================] - 1s 104us/step
# loss is : 0.08235542494400289
# scores is : 0.9787
```

#### keras入门实例之循环神经网络

> 使用简单的RNN进行手写数字分类

```python
from keras.models import Sequential
from keras.layers import Dense,Activation,SimpleRNN
import numpy as np 
np.random.seed(42)
from keras.utils import np_utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
%matplotlib inline

# 数据集获取 mnist
data = np.load("./data/mnist.npz")
X_train,y_train,X_test,y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']
# 此处注意一下数据集的维度：3维
X_train = X_train.reshape(-1,28,28) / 255 # 数据集归一化
X_test = X_test.reshape(-1,28,28) / 255 # 数据集归一化
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# 构建模型
model = Sequential()
# SimpleRNN 直接设定输出大小即可，SimpleRNN会根据输入的维度自动计算time_size和input_size
model.add(SimpleRNN(units=(50)))
# 输出层
model.add(Dense(10))
model.add(Activation("softmax"))

adam = Adam(lr = 1e-4)
model.compile(loss = 'categorical_crossentropy',
              optimizer=adam,
             metrics=['accuracy'])

print(" Training ---------------")
hist = model.fit(X_train,y_train,batch_size=32,epochs=6)
"""
 Training ---------------
Epoch 1/6
60000/60000 [==============================] - 34s 559us/step - loss: 1.5814 - acc: 0.4870
Epoch 2/6
60000/60000 [==============================] - 32s 538us/step - loss: 0.9048 - acc: 0.7312
Epoch 3/6
60000/60000 [==============================] - 33s 544us/step - loss: 0.6581 - acc: 0.8092
Epoch 4/6
60000/60000 [==============================] - 33s 555us/step - loss: 0.5315 - acc: 0.8458
Epoch 5/6
60000/60000 [==============================] - 33s 543us/step - loss: 0.4566 - acc: 0.8669
Epoch 6/6
60000/60000 [==============================] - 33s 558us/step - loss: 0.4075 - acc: 0.8803
"""
print(model.summary())
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_6 (SimpleRNN)     (None, 50)                3950      
_________________________________________________________________
dense_31 (Dense)             (None, 10)                510       
_________________________________________________________________
activation_49 (Activation)   (None, 10)                0         
=================================================================
Total params: 4,460
Trainable params: 4,460
Non-trainable params: 0
_________________________________________________________________
"""
print("Test -------------")
loss,accuray = model.evaluate(X_test,y_test)
print("loss is :",loss)
print("scores is :",accuray)
# Test -------------
# 10000/10000 [==============================] - 6s 568us/step
# loss is : 0.3777873183965683
# scores is : 0.8917
```

#### keras入门实例之自编码神经网络

> AutoEncoder无监督学习，提取特征

```python
# 自编码的无监督学习，降维提取图像的特征
from keras.models import Model
from keras.layers import Input,Dense
import numpy as np 
np.random.seed(42)
import matplotlib.pyplot as plt 
%matplotlib inline

# 数据集获取 mnist
data = np.load("./data/mnist.npz")
X_train,_,X_test,y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']
X_train = X_train.astype("float32") / 255 - 0.5
X_test = X_test.astype("float32") / 255 - 0.5
X_train = X_train.reshape(X_train.shape[0],-1) # 数据集归一化
X_test = X_test.reshape(X_test.shape[0],-1) # 数据集归一化

# 参数设置,提取的维度
encoder_dims = 2

# 函数式模型
# 定义输入层
input_img = Input(shape = (784,))
# 定义encode层
encoded = Dense(128,activation="relu")(input_img)
encoded = Dense(64,activation="relu")(encoded)
encoded = Dense(10,activation="relu")(encoded)
encoder_output = Dense(encoder_dims)(encoded)
# 定义decode层
decoded = Dense(10,activation="relu")(encoder_output)
decoded = Dense(64,activation="relu")(decoded)
decoded = Dense(128,activation="relu")(decoded)
decoder_output = Dense(784,activation="tanh")(decoded)

# 定义自编码模型
autoencoder = Model(input=input_img,output = decoder_output)
# 定义encode模型
encoder = Model(input=input_img,output=encoder_output)
# 编译模型
autoencoder.compile(loss = 'mse',optimizer = 'adam')

autoencoder.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 22        
_________________________________________________________________
dense_5 (Dense)              (None, 10)                30        
_________________________________________________________________
dense_6 (Dense)              (None, 64)                704       
_________________________________________________________________
dense_7 (Dense)              (None, 128)               8320      
_________________________________________________________________
dense_8 (Dense)              (None, 784)               101136    
=================================================================
Total params: 219,598
Trainable params: 219,598
Non-trainable params: 0
_________________________________________________________________
"""
# 训练模型
hist = autoencoder.fit(X_train,X_train,epochs=20,batch_size=32,shuffle=True)
"""
Epoch 1/20
60000/60000 [==============================] - 14s 240us/step - loss: 0.0546
Epoch 2/20
60000/60000 [==============================] - 13s 223us/step - loss: 0.0452
Epoch 3/20
60000/60000 [==============================] - 13s 219us/step - loss: 0.0428
Epoch 4/20
60000/60000 [==============================] - 13s 220us/step - loss: 0.0415
Epoch 5/20
60000/60000 [==============================] - 13s 218us/step - loss: 0.0408
Epoch 6/20
60000/60000 [==============================] - 14s 225us/step - loss: 0.0402
Epoch 7/20
60000/60000 [==============================] - 13s 221us/step - loss: 0.0397
Epoch 8/20
60000/60000 [==============================] - 13s 220us/step - loss: 0.0396
Epoch 9/20
60000/60000 [==============================] - 13s 224us/step - loss: 0.0396
Epoch 10/20
60000/60000 [==============================] - 13s 221us/step - loss: 0.0391
Epoch 11/20
60000/60000 [==============================] - 13s 224us/step - loss: 0.0390
Epoch 12/20
60000/60000 [==============================] - 13s 216us/step - loss: 0.0385
Epoch 13/20
60000/60000 [==============================] - 14s 230us/step - loss: 0.0385
Epoch 14/20
60000/60000 [==============================] - 14s 228us/step - loss: 0.0386
Epoch 15/20
60000/60000 [==============================] - 14s 226us/step - loss: 0.0383
Epoch 16/20
60000/60000 [==============================] - 14s 232us/step - loss: 0.0386
Epoch 17/20
60000/60000 [==============================] - 14s 232us/step - loss: 0.0381
Epoch 18/20
60000/60000 [==============================] - 14s 230us/step - loss: 0.0381
Epoch 19/20
60000/60000 [==============================] - 13s 223us/step - loss: 0.0379
Epoch 20/20
60000/60000 [==============================] - 14s 225us/step - loss: 0.0380
"""
# 降维
encoder_predict = encoder.predict(X_test)
# 可视化结果
plt.scatter(encoder_predict[:,0],encoder_predict[:,1],c=y_test)
plt.show()
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/encode_decode.png)

### 其他Keras使用细节

#### 指定占用的GPU

> keras在使用GPU的时候有个特点，就是默认全部占满显存。 这样如果有多个模型都需要使用GPU跑的话，那么限制是很大的，而且对于GPU也是一种浪费。因此在使用keras时需要有意识的设置运行时使用那块显卡，需要使用多少容量。
>
> 这方面的设置一般有三种情况： 
>
> - 指定显卡 
> - 限制GPU用量 
> - 即指定显卡又限制GPU用量

##### 指定显卡

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

##### 限制GPU用量 

```python
# 如果出现显存的问题,那么需要使用以下的代码
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' # 分配策略
config.gpu_options.per_process_gpu_memory_fraction = 0.3 # 显存的占比
config.gpu_options.allow_growth = True # 允许显存增量使用
set_session(tf.Session(config=config)) 
```

##### 指定显卡和限制GPU用量

```python
# 也就是将上面的两种情况结合起来使用
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)
```

#### 查看与保存模型结构

##### 查看搭建的网络

```python
model.summary()

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_6 (SimpleRNN)     (None, 50)                3950      
_________________________________________________________________
dense_31 (Dense)             (None, 10)                510       
_________________________________________________________________
activation_49 (Activation)   (None, 10)                0         
=================================================================
Total params: 4,460
Trainable params: 4,460
Non-trainable params: 0
_________________________________________________________________
"""
```

##### 图片的方式保存模型的结构

```python
from keras.utils.vis_utils import plot_model

plot_model(autoencoder, to_file='./data/autoEncode.png')
```

### 参考链接

> - [keras中文文档（官方）](https://keras.io/zh/)
> - [keras中文文档（非官方）](http://keras-cn.readthedocs.io/en/latest/)
> - [莫烦keras教程代码](https://github.com/MorvanZhou/tutorials/tree/master/kerasTUT)
> - [莫烦keras视频教程](https://www.bilibili.com/video/av16910214/)
> - [Keras FAQ: Frequently Asked Keras Questions](https://github.com/keras-team/keras/blob/1d8121f9ff8a8f32df99004c33674907c8919602/docs/templates/getting-started/faq.md#is-the-data-shuffled-during-training)
> - [一个不负责任的Keras介绍（上）](https://zhuanlan.zhihu.com/p/22129946)
> - [一个不负责任的Keras介绍（中）](https://zhuanlan.zhihu.com/p/22129301)
> - [一个不负责任的Keras介绍（下）](https://zhuanlan.zhihu.com/p/22135796)
