# Pytorch快速入门0


### 为什么选择`PyTorch`

> - 简洁：PyTorch的设计追求最少的封装，尽量避免重复造轮子。不像TensorFlow中充斥着`session、graph、operation、name_scope、variable、tensor、layer`等全新的概念，PyTorch的设计遵循`tensor→autograd→nn.Module `三个由低到高的抽象层次，分别代表高维数组（张量）、自动求导（变量）和神经网络（层/模块），而且这三个抽象之间联系紧密，可以同时进行修改和操作。
> - 速度：PyTorch的灵活性不以速度为代价，在许多评测中，PyTorch的速度表现胜过TensorFlow和Keras等框架 。框架的运行速度和程序员的编码水平有极大关系，但同样的算法，使用PyTorch实现的那个更有可能快过用其他框架实现的。
> - 易用：PyTorch是所有的框架中面向对象设计的最优雅的一个。PyTorch的面向对象的接口设计来源于Torch，而Torch的接口设计以灵活易用而著称，Keras作者最初就是受Torch的启发才开发了Keras。PyTorch继承了Torch的衣钵，尤其是API的设计和模块的接口都与Torch高度一致。PyTorch的设计最符合人们的思维，它让用户尽可能地专注于实现自己的想法，即所思即所得，不需要考虑太多关于框架本身的束缚。
> - 活跃的社区：PyTorch提供了完整的文档，循序渐进的指南，作者亲自维护的论坛 供用户交流和求教问题。Facebook 人工智能研究院对PyTorch提供了强力支持，作为当今排名前三的深度学习研究机构，FAIR的支持足以确保PyTorch获得持续的开发更新，不至于像许多由个人开发的框架那样昙花一现。
>
> <!--more -->
>
> PyTorch还有一个优点就是**Torch**自称为神经网络界的**Numpy**，它能将**torch**产生的**tensor**放在GPU中加速运算，就想Numpy会把array放在CPU中加速运算。所以在神经网络中，用Torch的tensor形式更优。我们可以把Pytorch当做Numpy来用。
>
> PyTorch使用的是动态图，它的计算图在每次前向传播时都是从头开始构建，所以它能够使用Python控制语句（如for、if等）根据需求创建计算图。这点在自然语言处理领域中很有用，它意味着你不需要事先构建所有可能用到的图的路径，图在运行时才构建。

### `PyTorch`的安装

> - pip安装方式
> - conda安装方式

```shell
# win+python3.6
pip3 install https://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-win_amd64.whl
pip3 install torchvision
# win+python3.6+conda
conda install pytorch torchvision cuda80 -c pytorch
```

[更多方法参见此处](https://pytorch.org/get-started/locally/)

### `PyTorch`的核心概念

#### `Tensor`：张量

> Tensor是PyTorch中重要的数据结构，可认为是一个高维数组。它可以是一个数（标量）、一维数组（向量）、二维数组（矩阵）以及更高维的数组。
>
> Tensor和Numpy的ndarrays类似，但Tensor可以使用GPU进行加速。Tensor的使用和Numpy及Matlab的接口十分相似。
>
> `torch.Tensor`是一种包含单一数据类型元素的多维矩阵。

##### `Tensor`属性

> 每个`torch.Tensor`都有`torch.dtype`, `torch.device`,和`torch.layout`。

###### `torch.dtype`

> Torch定义了七种CPU张量类型和八种GPU张量类型：

| Data tyoe                | CPU tensor           | GPU tensor                |
| ------------------------ | -------------------- | ------------------------- |
| 32-bit floating point    | `torch.FloatTensor`  | `torch.cuda.FloatTensor`  |
| 64-bit floating point    | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
| 16-bit floating point    | N/A                  | `torch.cuda.HalfTensor`   |
| 8-bit integer (unsigned) | `torch.ByteTensor`   | `torch.cuda.ByteTensor`   |
| 8-bit integer (signed)   | `torch.CharTensor`   | `torch.cuda.CharTensor`   |
| 16-bit integer (signed)  | `torch.ShortTensor`  | `torch.cuda.ShortTensor`  |
| 32-bit integer (signed)  | `torch.IntTensor`    | `torch.cuda.IntTensor`    |
| 64-bit integer (signed)  | `torch.LongTensor`   | `torch.cuda.LongTensor`   |

###### `torch.device`

> - `torch.device`代表将`torch.Tensor`分配到的设备的对象。
> - `torch.device`包含一个设备类型（`'cpu'`或`'cuda'`设备类型）和可选的设备的序号。如果设备序号不存在，则为当前设备; 例如，`torch.Tensor`用设备构建`'cuda'`的结果等同于`'cuda:X'`,其中`X`是`torch.cuda.current_device()`的结果。
> - `torch.Tensor`的设备可以通过`Tensor.device`访问属性。
> - 构造`torch.device`可以通过字符串/字符串和设备编号。

```python
torch.device('cuda:0')
# device(type='cuda', index=0)
torch.device('cpu')
# device(type='cpu')
torch.device('cuda', 0)
# device(type='cuda', index=0)
```

> **注意**
> `torch.device`函数中的参数通常可以用一个字符串替代。这允许使用代码快速构建原型。

```python
# Example of a function that takes in a torch.device
cuda1 = torch.device('cuda:1')
torch.randn((2,3), device=cuda1)
# You can substitute the torch.device with a string
torch.randn((2,3), 'cuda:1')
```

###### `torch.layout`

> - `torch.layout`表示`torch.Tensor`内存布局的对象。目前，我们支持`torch.strided(dense Tensors)`并为`torch.sparse_coo(sparse COO Tensors)`提供实验支持。
> - `torch.strided`代表密集张量，是最常用的内存布局。每个`strided`张量都会关联 一个`torch.Storage`，它保存着它的数据。这些张力提供了多维度， 存储的`strided`视图。`Strides`是一个整数型列表：`k-th stride`表示在张量的第k维从一个元素跳转到下一个元素所需的内存。这个概念使得可以有效地执行多张量。

```python
x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
x.stride()
# (5, 1)

x.t().stride()
# (1, 5)
```

##### `Tensor`方法

> Tensor的方法中，带有`_`的方法代表能够修改Tensor本身。比如，`torch.FloatTensor.abs_()`会在原地计算绝对值并返回修改的张量，而`tensor.FloatTensor.abs()`将会在新张量中计算结果。

###### `Tensor.copy_(src, async=False)`

> 将`src`中的元素复制到tensor中并返回这个tensor。 如果broadcast是True，则源张量必须可以使用该张量广播。否则两个tensor应该有相同数目的元素，可以是不同的数据类型或存储在不同的设备上。
>
> 参数：
>
> - src（Tensor） - 要复制的源张量
> - async（bool） - 如果为True，并且此副本位于CPU和GPU之间，则副本可能会相对于主机异步发生。对于其他副本，此参数无效。
> - broadcast（bool） - 如果为True，src将广播到底层张量的形状。

###### `Tensor.cuda(device=None, async=False)`

######  

> 返回此对象在CPU内存中的一个副本 如果该对象已经在CUDA内存中，并且在正确的设备上，则不会执行任何副本，并返回原始对象。
>
> 参数：
>
> - device（int） ：目标GPU ID。默认为当前设备。
> - async（bool） ：如果为True并且源处于固定内存中，则该副本将相对于主机是异步的。否则，该参数没有意义。

###### `Tensor.expand(*sizes)`

> 返回tensor的一个新视图，单个维度扩大为更大的尺寸。 tensor也可以扩大为更高维，新增加的维度将附在前面。 扩大tensor不需要分配新内存，只是仅仅新建一个tensor的视图，其中通过将`stride`设为0，一维将会扩展位更高维。任何一个一维的在不分配新内存情况下可扩展为任意的数值。
>
> 参数：
>
> - sizes(torch.Size or int...)-需要扩展的大小

###### `Tensor.narrow(*dimension, start, length*)`

> 返回这个张量的缩小版本的新张量。维度`dim`缩小范围是`start`到`start+length`。返回的张量和该张量共享相同的底层存储。
>
> 参数：
>
> - dimension (*int*)-需要缩小的维度
> - start (*int*)-起始维度
> - length (*int*)-长度

###### `Tensor.resize_(*sizes)`

> 将tensor的大小调整为指定的大小。如果元素个数比当前的内存大小大，就将底层存储大小调整为与新元素数目一致的大小。如果元素个数比当前内存小，则底层存储不会被改变。原来tensor中被保存下来的元素将保持不变，但新内存将不会被初始化。
>
> 参数：
>
> - sizes (torch.Size or int...)-需要调整的大小

[更多的方法参考此处](https://ptorch.com/docs/8/torch-tensor)

下面通过一些实例学习，Tensor的使用。

```python
# 构建 5x3 矩阵，只是分配了空间，未初始化
x = t.Tensor(5, 3)
x = t.Tensor([[1,2],[3,4]])
# tensor([[ 1.,  2.],
#         [ 3.,  4.]])
# 使用[0,1]均匀分布随机初始化二维数组
x = t.rand(5, 3)  
# tensor([[ 0.8052,  0.7188,  0.0332],
#         [ 0.6054,  0.8955,  0.8972],
#         [ 0.1107,  0.3319,  0.0336],
#         [ 0.2394,  0.5188,  0.2201],
#         [ 0.9730,  0.9370,  0.5677]])
x.size()[1], x.size(1) # 查看列的个数, 两种写法等价
# (3,3)
```

> `torch.Size` 是tuple对象的子类，因此它支持tuple的所有操作，如x.size()[0]

```python
# 加减乘除运算
y = t.rand(5, 3)
# 加法的第一种写法
x + y
# tensor([[ 0.9639,  0.8763,  0.2834],
#         [ 1.3785,  1.5090,  1.3919],
#         [ 0.7139,  0.6348,  0.8439],
#         [ 0.7022,  1.5079,  0.4776],
#         [ 1.7892,  1.6383,  0.7774]])
# 加法的第二种写法
t.add(x, y)
# tensor([[ 0.9639,  0.8763,  0.2834],
#         [ 1.3785,  1.5090,  1.3919],
#         [ 0.7139,  0.6348,  0.8439],
#         [ 0.7022,  1.5079,  0.4776],
#         [ 1.7892,  1.6383,  0.7774]])
# 加法的第三种写法：指定加法结果的输出目标为result
result = t.Tensor(5, 3) # 预先分配空间
t.add(x, y, out=result) # 输入到result
# tensor([[ 0.9639,  0.8763,  0.2834],
#         [ 1.3785,  1.5090,  1.3919],
#         [ 0.7139,  0.6348,  0.8439],
#         [ 0.7022,  1.5079,  0.4776],
#         [ 1.7892,  1.6383,  0.7774]])

print('最初y')
print(y)
# tensor([[ 0.1587,  0.1575,  0.2501],
#         [ 0.7732,  0.6135,  0.4947],
#         [ 0.6033,  0.3029,  0.8103],
#         [ 0.4628,  0.9891,  0.2575],
#         [ 0.8163,  0.7013,  0.2097]])

print('第一种加法，y的结果')
y.add(x) # 普通加法，不改变y的内容
print(y)
# 第一种加法，y的结果
# tensor([[ 0.1587,  0.1575,  0.2501],
#         [ 0.7732,  0.6135,  0.4947],
#         [ 0.6033,  0.3029,  0.8103],
#         [ 0.4628,  0.9891,  0.2575],
#         [ 0.8163,  0.7013,  0.2097]])

print('第二种加法，y的结果')
y.add_(x) # inplace 加法，y变了
print(y)
# 第二种加法，y的结果
# tensor([[ 0.9639,  0.8763,  0.2834],
#         [ 1.3785,  1.5090,  1.3919],
#         [ 0.7139,  0.6348,  0.8439],
#         [ 0.7022,  1.5079,  0.4776],
#         [ 1.7892,  1.6383,  0.7774]])
```

> 注意，函数名后面带下划线**`_`** 的函数会修改Tensor本身。例如，`x.add_(y)`和`x.t_()`会改变 `x`，但`x.add(y)`和`x.t()`返回一个新的Tensor， 而`x`不变。
>
> Tensor还支持很多操作，包括数学运算、线性代数、选择、切片等等，其接口设计与Numpy极为相似。更详细的使用方法。
>
> Tensor和numpy对象共享内存，所以他们之间的转换很快，而且几乎不会消耗什么资源。但这也意味着，如果其中一个变了，另外一个也会随之改变。
>
> Tensor和Numpy的数组之间的互操作非常容易且快速。对于Tensor不支持的操作，可以先转为Numpy数组处理，之后再转回Tensor。

```python
a = t.ones(5) # 新建一个全1的Tensor
# tensor([ 1.,  1.,  1.,  1.,  1.])
b = a.numpy() # Tensor -> Numpy
# array([1., 1., 1., 1., 1.], dtype=float32)

a = np.ones(5)
b = t.from_numpy(a) # Numpy->Tensor
print(a)
# [1. 1. 1. 1. 1.]
print(b) 
# tensor([ 1.,  1.,  1.,  1.,  1.], dtype=torch.float64)

# 共享内存，修改numpy会修改tensor
b.add_(1) # 以`_`结尾的函数会修改自身
print(a)
# [2. 2. 2. 2. 2.]
print(b) # Tensor和Numpy共享内存
# tensor([ 2.,  2.,  2.,  2.,  2.], dtype=torch.float64)
```

> 如果你想获取某一个元素的值，可以使用`scalar.item`。 直接`tensor[idx]`得到的还是一个tensor: 一个0-dim 的tensor，一般称为scalar.

```python
scalar = b[0]
# tensor(2., dtype=torch.float64)
scalar.size() #0-dim
# torch.Size([])
scalar.item() # 使用scalar.item()能从中取出python对象的数值
# 2.0
tensor = t.tensor([2]) # 注意和scalar的区别
tensor,scalar
# (tensor([ 2]), tensor(2., dtype=torch.float64))
tensor.size(),scalar.size()
# (torch.Size([1]), torch.Size([]))
# 只有一个元素的tensor也可以调用`tensor.item()`
tensor.item(), scalar.item()
# (2, 2.0)
```

> PyTorch中还有一个和`np.array` 很类似的接口: `torch.tensor`, 二者的使用十分类似。
>
> 需要注意的是，`t.tensor()`总是会进行数据拷贝，新tensor和原来的数据不再共享内存。所以如果你想共享内存的话，建议使用`torch.from_numpy()`或者`tensor.detach()`来新建一个tensor, 二者共享内存。
>
> Tensor可通过`.cuda` 方法转为GPU的Tensor，从而享受GPU带来的加速运算。

```python
tensor = t.tensor([3,4]) # 新建一个包含 3，4 两个元素的tensor
# 以下可以看到，新建的tensor与原来的数据不共享内存
old_tensor = tensor
new_tensor = t.tensor(old_tensor)
new_tensor[0] = 1111
# old_tensor, new_tensor
# (tensor([ 3,  4]), tensor([ 1111,     4]))

# 以下使用detach新建tensor会共享内存
new_tensor = old_tensor.detach()
new_tensor[0] = 1111
# old_tensor, new_tensor
# (tensor([ 1111,     4]), tensor([ 1111,     4]))
```

##### 详解`Tensor`操作

> 接口的角度来讲，对tensor的操作可分为两类：
>
> 1. `torch.function`，如`torch.save`等。
> 2. 另一类是`tensor.function`，如`tensor.view`等。
>
> 为方便使用，对tensor的大部分操作同时支持这两类接口，
>
> 而从存储的角度来讲，对tensor的操作又可分为两类：
>
> 1. 不会修改自身的数据，如 `a.add(b)`， 加法的结果会返回一个新的tensor。
> 2. 会修改自身的数据，如 `a.add_(b)`， 加法的结果仍存储在a中，a被修改了。
>
> 函数名以`_`结尾的都是inplace方式, 即会修改调用者自己的数据，在实际应用中需加以区分。

###### 创建Tensor

> 在PyTorch中新建tensor的方法有很多，具体可以参见下表。

 常见新建tensor的方法

|                函数                 |        功能        |
| :-------------------------------: | :--------------: |
|          Tensor(\*sizes)          |      基础构造函数      |
|           tensor(data,)           | 类似np.array的构造函数  |
|           ones(\*sizes)           |     全1Tensor     |
|          zeros(\*sizes)           |     全0Tensor     |
|           eye(\*sizes)            |    对角线为1，其他为0    |
|          arange(s,e,step          |   从s到e，步长为step   |
|        linspace(s,e,steps)        | 从s到e，均匀切分成steps份 |
|        rand/randn(\*sizes)        |     均匀/标准分布      |
| normal(mean,std)/uniform(from,to) |    正态分布/均匀分布     |
|            randperm(m)            |       随机排列       |

> 这些创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu).
>
> 其中使用`Tensor`函数新建tensor是最复杂多变的方式，它既可以接收一个list，并根据list的数据新建tensor，也能根据指定的形状新建tensor，还能传入其他的tensor。
>
> PS:`t.Tensor(*sizes)`创建tensor时，系统不会马上分配空间，只是会计算剩余的内存是否足够使用，使用到tensor时才会分配，而其它操作都是在创建完tensor之后马上进行空间分配。

```python
# 1.指定tensor的形状
a = t.Tensor(2, 3)
# 2.用list的数据创建tensor
b = t.Tensor([[1,2,3],[4,5,6]])
b_size = b.size()
b.tolist() # 把tensor转为list
b.numel() # b中元素总个数，2*3，等价于b.nelement()
# 3.创建一个和b形状一样的tensor
c = t.Tensor(b_size)
# 4.创建一个元素为2和3的tensor
d = t.Tensor((2, 3))
# 其他的新建Tensor的操作
t.ones(2, 3) # 全1
t.zeros(2, 3) #　全0
t.arange(1, 6, 2) #　生成序列
t.linspace(1, 10, 3) # 生成序列
t.randn(2, 3, device=t.device('cpu')) # 生成随机数
t.randperm(5) # 长度为5的随机排列
t.eye(2, 3, dtype=t.int) # 对角线为1, 不要求行列数一致

# torch.tensor()是新增加的函数，使用的方法，和参数几乎和`np.array`完全一致
scalar = t.tensor(3.14159) 
print('scalar: %s, shape of sclar: %s' %(scalar, scalar.shape))
#　scalar: tensor(3.1416), shape of sclar: torch.Size([])
t.tensor([[0.11111, 0.222222, 0.3333333]],
                     dtype=t.float64,
                     device=t.device('cpu'))
# tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64)
empty_tensor = t.tensor([])
empty_tensor.shape
# torch.Size([0])
```

###### Tensor的基本操作

> 通过`tensor.view`方法可以调整tensor的形状，但必须保证调整前后元素总数一致。`view`不会修改自身的数据，返回的新tensor与源tensor共享内存，也即更改其中的一个，另外一个也会跟着改变。
>
> 在实际应用中可能经常需要添加或减少某一维度，这时候`squeeze`和`unsqueeze`两个函数就派上用场了。`squeeze`降维，`unsqueeze`升维。
>
> `resize`是另一种可用来调整`size`的方法，但与`view`不同，它可以修改tensor的大小。如果新大小超过了原大小，会自动分配新的内存空间，而如果新大小小于原大小，则之前的数据依旧会被保存。

```python
import torch as t 

a = t.arange(0, 6)
a.view(2, 3)
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.]])
b = a.view(-1, 3) # 当某一维为-1的时候，会自动计算它的大小
b.shape
# torch.Size([2, 3])
b.unsqueeze(1) # 注意形状，在第1维（下标从0开始）上增加“１” 维
#等价于 b[:,None]
b[:, None].shape
# torch.Size([2, 1, 3])
b.unsqueeze(-2) # -2表示倒数第二个维度上面增加“1’维
c = b.view(1, 1, 1, 2, 3)
c.shape
# torch.Size([1, 1, 1, 2, 3])
c.squeeze_(0) # 压缩第0维的“１”
c.shape
# torch.Size([1, 1, 2, 3])
c.squeeze() # 把所有维度为“1”的压缩降维
# view之后和原来的数据共享
a[1] = 100
b # a修改，b作为view之后的，也会跟着修改
# tensor([[   0.,  100.,    2.],
#         [   3.,    4.,    5.]])
# reseize尺寸小于原尺寸，部分数据会被保留，不显示
b.resize_(1, 3)
b
# tensor([[   0.,  100.,    2.]])
# resize尺寸大于原尺寸，如果有隐藏数据会显示，其他多出的大小会被分配新空间
b.resize_(3, 3) # 旧的数据依旧保存着，多出的大小会分配新空间
b
# tensor([[   0.0000,  100.0000,    2.0000],
#         [   3.0000,    4.0000,    5.0000],
#         [  -0.0000,    0.0000,    0.0000]])
```

###### Tensor索引操作

> Tensor支持与numpy.ndarray类似的索引操作，语法上也类似。

###### Tensor元素操作

> 这部分操作会对tensor的每一个元素(point-wise，又名element-wise)进行操作，此类操作的输入与输出形状一致。常用的操作如下表所示。

常见的逐元素操作

|               函数                |          功能           |
| :-----------------------------: | :-------------------: |
| abs/sqrt/div/exp/fmod/log/pow.. | 绝对值/平方根/除法/指数/求余/求幂.. |
|    cos/sin/asin/atan2/cosh..    |        相关三角函数         |
|     ceil/round/floor/trunc      | 上取整/四舍五入/下取整/只保留整数部分  |
|     clamp(input, min, max)      |     超过min和max部分截断     |
|          sigmod/tanh..          |         激活函数          |

对于很多操作，例如div、mul、pow、fmod等，PyTorch都实现了运算符重载，所以可以直接使用运算符。如`a ** 2` 等价于`torch.pow(a,2)`, `a * 2`等价于`torch.mul(a,2)`。

其中`clamp(x, min, max)`的输出满足以下公式：

$$
y_i =
\begin{cases}
min,  & \text{if  } x_i \lt min \\
x_i,  & \text{if  } min \le x_i \le max  \\
max,  & \text{if  } x_i \gt max\\
\end{cases}
$$

`clamp`常用在某些需要比较大小的地方，如取一个tensor的每个元素与另一个数的较大值。

```python
import torch as t 

# 生成数据
a = t.arange(0, 6).view(2, 3)
t.cos(a)
# tensor([[1.0000000000, 0.5403022766, -0.4161468446],
#         [-0.9899924994, -0.6536436081, 0.2836622000]])
a % 3 # 等价于t.fmod(a, 3)
# tensor([[ 0.,  1.,  2.],
#         [ 0.,  1.,  2.]])
t.clamp(a, min=3)
# tensor([[ 3.,  3.,  3.],
#         [ 3.,  4.,  5.]])
```

###### Tensor归并操作

> 此类操作会使输出形状小于输入形状，并可以沿着某一维度进行指定操作。如加法`sum`，既可以计算整个tensor的和，也可以计算tensor中每一行或每一列的和。常用的归并操作如下表所示。

常用归并操作

|          函数          |     功能      |
| :------------------: | :---------: |
| mean/sum/median/mode | 均值/和/中位数/众数 |
|      norm/dist       |    范数/距离    |
|       std/var        |   标准差/方差    |
|    cumsum/cumprod    |    累加/累乘    |

> 以上大多数函数都有一个参数**`dim`**，用来指定这些操作是在哪个维度上执行的。关于`dim`(对应于`Numpy`中的`axis`)的解释众说纷纭，这里提供一个简单的记忆方式：
>
> 假设输入的形状是(m, n, k)
>
> - 如果指定dim=0，输出的形状就是(1, n, k)或者(n, k)
> - 如果指定dim=1，输出的形状就是(m, 1, k)或者(m, k)
> - 如果指定dim=2，输出的形状就是(m, n, 1)或者(m, n)
>
> size中是否有"1"，取决于参数`keepdim`，`keepdim=True`会保留维度`1`。注意，以上只是经验总结，并非所有函数都符合这种形状变化方式，如`cumsum`。

```python
import torch as t 

# 生成数据
b = t.ones(2, 3)
b.sum(dim = 0, keepdim=True)
# tensor([[ 2.,  2.,  2.]])
# keepdim=False，不保留维度"1"，注意形状
b.sum(dim=0, keepdim=False)
# tensor([ 2.,  2.,  2.])
```

###### Tensor比较操作

> 比较函数中有一些是逐元素比较，操作类似于逐元素操作，还有一些则类似于归并操作。常用比较函数如下表所示。

常用比较函数

|        函数         |          功能           |
| :---------------: | :-------------------: |
| gt/lt/ge/le/eq/ne | 大于/小于/大于等于/小于等于/等于/不等 |
|       topk        |        最大的k个数         |
|       sort        |          排序           |
|      max/min      |    比较两个tensor最大最小值    |

> 表中第一行的比较操作已经实现了运算符重载，因此可以使用`a>=b`、`a>b`、`a!=b`、`a==b`，其返回结果是一个`ByteTensor`，可用来选取元素。`max/min`这两个操作比较特殊，以max来说，它有以下三种使用情况：
>
> - t.max(tensor)：返回tensor中最大的一个数
> - t.max(tensor,dim)：指定维上最大的数，返回tensor和下标
> - t.max(tensor1, tensor2): 比较两个tensor相比较大的元素
>
> 至于比较一个tensor和一个数，可以使用clamp函数。

```python
import torch as t 

# 生成数据a
a = t.linspace(0, 15, 6).view(2, 3)
a
# tensor([[  0.,   3.,   6.],
#         [  9.,  12.,  15.]])
# 生成数据b
b = t.linspace(15, 0, 6).view(2, 3)
b
# tensor([[ 15.,  12.,   9.],
#         [  6.,   3.,   0.]])
a[a>b] # a中大于b的元素
# tensor([  9.,  12.,  15.])
t.max(b, dim=1) 
# 第一个返回值的15和6分别表示第0行和第1行最大的元素
# 第二个返回值的0和0表示上述最大的数是该行第0个元素
# (tensor([ 15.,   6.]), tensor([ 0,  0]))
```

###### Tensor线性代数

> `PyTorch`的线性函数主要封装了`Blas`和`Lapack`，其用法和接口都与`Numpy`类似。常用的线性代数函数如下表所示。

常用的线性代数函数

|                函数                |        功能         |
| :------------------------------: | :---------------: |
|              trace               |   对角线元素之和(矩阵的迹)   |
|               diag               |       对角线元素       |
|            triu/tril             | 矩阵的上三角/下三角，可指定偏移量 |
|              mm/bmm              |  矩阵乘法，batch的矩阵乘法  |
| addmm/addbmm/addmv/addr/badbmm.. |       矩阵运算        |
|                t                 |        转置         |
|            dot/cross             |       内积/外积       |
|             inverse              |       求逆矩阵        |
|               svd                |       奇异值分解       |

> 具体使用说明请参见[官方文档](http://pytorch.org/docs/torch.html#blas-and-lapack-operations)，需要注意的是，矩阵的转置会导致存储空间不连续，需调用它的`.contiguous`方法将其转为连续。

###### Tensor广播法则

> 广播法则(`broadcast`)是科学运算中经常使用的一个技巧，它在快速执行向量化的同时不会占用额外的内存/显存。
> `Numpy`的广播法则定义如下：
>
> - 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分通过在前面加1补齐
> - 两个数组要么在某一个维度的长度一致，要么其中一个为1，否则不能计算 
> - 当输入数组的某个维度的长度为1时，计算时沿此维度复制扩充成一样的形状
>
> PyTorch当前已经支持了自动广播法则，但是通过以下两个函数的组合手动实现广播法则，这样更直观，更不易出错：
>
> - `unsqueeze`或者`view`，或者`tensor[None]`,：为数据某一维的形状补1，实现法则1
> - `expand`或者`expand_as`，重复数组，实现法则3；该操作不会复制数组，所以不会占用额外的空间。
>
> 注意，`repeat`实现与`expand`相类似的功能，但是repeat会把相同数据复制多份，因此会占用额外的空间。

```python
import torch as t

# 生成数据a和b
a = t.ones(3, 2)
b = t.zeros(2, 3,1)
# 自动广播法则
# 第一步：a是2维,b是3维，所以先在较小的a前面补1 ，
#               即：a.unsqueeze(0)，a的形状变成（1，3，2），b的形状是（2，3，1）,
# 第二步:   a和b在第一维和第三维形状不一样，其中一个为1 ，
#               可以利用广播法则扩展，两个形状都变成了（2，3，2）
a+b
# tensor([[[ 1.,  1.],
#          [ 1.,  1.],
#          [ 1.,  1.]],

#         [[ 1.,  1.],
#          [ 1.,  1.],
#          [ 1.,  1.]]])
# 手动广播法则
# 或者 a.view(1,3,2).expand(2,3,2) + b.expand(2,3,2)
a[None].expand(2, 3, 2) + b.expand(2,3,2)
# tensor([[[ 1.,  1.],
#          [ 1.,  1.],
#          [ 1.,  1.]],

#         [[ 1.,  1.],
#          [ 1.,  1.],
#          [ 1.,  1.]]])
# expand不会占用额外空间，只会在需要的时候才扩充，可极大节省内存
e = a.unsqueeze(0).expand(10000000000000, 3,2)
```

##### Tensor内部结构

> `tensor`的数据结构如下图所示。tensor分为头信息区(Tensor)和存储区(Storage)，信息区主要保存着tensor的形状（size）、步长（stride）、数据类型（type）等信息，而真正的数据则保存成连续数组。由于数据动辄成千上万，因此信息区元素占用内存较少，主要内存占用则取决于tensor中元素的数目，也即存储区的大小。
>
> 一般来说一个tensor有着与之相对应的storage, storage是在data之上封装的接口，便于使用，而不同tensor的头信息一般不同，但却可能使用相同的数据。
>
> ![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/tensor_data_structure.svg)

```python
import torch as t 
# 生成数据a
a = t.arange(0, 6)
# 查看a的storage
a.storage()
#  0.0
#  1.0
#  2.0
#  3.0
#  4.0
#  5.0
# [torch.FloatStorage of size 6]
# b由a view生成
b = a.view(2, 3)
# 查看b的storage
b.storage()
#  0.0
#  1.0
#  2.0
#  3.0
#  4.0
#  5.0
# [torch.FloatStorage of size 6]

# 一个对象的id值可以看作它在内存中的地址
# storage的内存地址一样，即是同一个storage
id(b.storage()) == id(a.storage())
# True

# a改变，b也随之改变，因为他们共享storage
a[1] = 100
b
# tensor([[   0.,  100.,    2.],
#         [   3.,    4.,    5.]])
# c由a切片得到
c = a[2:] 
c.storage()
# 0.0
# 100.0
# 2.0
# 3.0
# 4.0
# 5.0
# [torch.FloatStorage of size 6]
c.data_ptr(), a.data_ptr() # data_ptr返回tensor首元素的内存地址
# 可以看出相差8，这是因为2*4=8--相差两个元素，每个元素占4个字节(float)
# (93894489135160, 93894489135152)
c[0] = -100 # c[0]的内存地址对应a[2]的内存地址
a
# tensor([   0.,  100., -100.,    3.,    4.,    5.])
# 使用storage来初始化Tensor
d = t.Tensor(c.storage())
d[0] = 6666
b
# tensor([[ 6666.,   100.,  -100.],
#         [    3.,     4.,     5.]])
# 下面４个tensor共享storage
id(a.storage()) == id(b.storage()) == id(c.storage()) == id(d.storage())
# True
a.storage_offset(), c.storage_offset(), d.storage_offset()
# (0, 2, 0)
# 切片得到e
e = b[::2, ::2] # 隔2行/列取一个元素
id(e.storage()) == id(a.storage())
# True
b.stride(), e.stride()
# ((3, 1), (6, 2))
# e的数据变得不连续
e.is_contiguous()
# False
```

> 可见绝大多数操作并不修改`tensor`的数据，而只是修改了`tensor`的头信息。这种做法更节省内存，同时提升了处理速度。在使用中需要注意。
> 此外有些操作会导致`tensor`不连续，这时需调用`tensor.contiguous`方法将它们变成连续的数据，该方法会使数据复制一份，不再与原来的数据共享`storage`。

##### 其他`Tensor`使用技巧

###### GPU/CPU

> `tensor`可以很随意的在`gpu/cpu`上传输。使用`tensor.cuda(device_id)`或者`tensor.cpu()`。另外一个更通用的方法是`tensor.to(device)`。

```python
import torch as t
# 生成数据
a = t.randn(3, 4)
a.device
# device(type='cpu')
device = t.device('gpu')
a.to(device)
```

> **注意**
>
> - 尽量使用`tensor.to(device)`, 将`device`设为一个可配置的参数，这样可以很轻松的使程序同时兼容GPU和CPU
> - 数据在GPU之中传输的速度要远快于内存(CPU)到显存(GPU), 所以尽量避免频繁的在内存和显存中传输数据。

###### 持久化

> `Tensor`的保存和加载十分的简单，使用t.save和t.load即可完成相应的功能。在save/load时可指定使用的`pickle`模块，在load时还可将GPU tensor映射到CPU或其它GPU上。

```python
if t.cuda.is_available():
    a = a.cuda(1) # 把a转为GPU1上的tensor,
    t.save(a,'a.pth')

    # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
    b = t.load('a.pth')
    # 加载为c, 存储于CPU
    c = t.load('a.pth', map_location=lambda storage, loc: storage)
    # 加载为d, 存储于GPU0上
    d = t.load('a.pth', map_location={'cuda:1':'cuda:0'})
```

#### `autograd`：自动微分

> 深度学习的算法本质上是通过反向传播求导数，而PyTorch的**`autograd`**模块则实现了此功能。在Tensor上的所有操作，**`autograd`**都能为它们自动提供微分，避免了手动计算导数的复杂过程。
>
> ~~`autograd.Variable`是Autograd中的核心类，它简单封装了Tensor，并支持几乎所有Tensor有的操作。Tensor在被封装为Variable之后，可以调用它的`.backward`实现反向传播，自动计算所有梯度~~ ~~Variable的数据结构如下图所示。~~
>
> ![图2-6:Variable的数据结构](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/autograd_Variable.svg)
>
>   *从0.4起, Variable 正式合并入Tensor, Variable 本来实现的自动微分功能，Tensor就能支持。读者还是可以使用Variable(tensor), 但是这个操作其实什么都没做。所以以后可以直接使用tensor，而不是Variable*. 
>
>   要想使得Tensor使用autograd功能，只需要设置`tensor.requries_grad=True`. 
>
> ~~Variable主要包含三个属性。~~
> ~~- `data`：保存Variable所包含的Tensor~~
> ~~- `grad`：保存`data`对应的梯度，`grad`也是个Variable，而不是Tensor，它和`data`的形状一样。~~
> ~~- `grad_fn`：指向一个`Function`对象，这个`Function`用来反向传播计算输入的梯度。~~

##### `Variable`

> `autograd`中的核心数据结构是`Variable`。从v0.4版本起，`Variable`和`Tensor`合并。我们可以认为需要求导(requires_grad)的tensor即Variable。autograd记录对tensor的操作记录用来构建计算图。
>
> Variable提供了大部分tensor支持的函数，但其不支持部分`inplace`函数，因这些函数会修改tensor自身，而在反向传播中，variable需要缓存原来的tensor来计算反向传播梯度。如果想要计算各个Variable的梯度，只需调用根节点variable的`backward`方法，autograd会自动沿着计算图反向传播，计算每一个叶子节点的梯度。
>
> `Tensor.backward(gradient=None, retain_graph=None, create_graph=None)`主要有如下参数：
>
> - gradient：形状与variable一致，对于`y.backward()`，grad_variables相当于链式法则$${dz \over dx}={dz \over dy} \times {dy \over dx}$$中的$$\textbf {dz} \over \textbf {dy}$$。grad_variables也可以是tensor或序列。
> - retain_graph：反向传播需要缓存一些中间结果，反向传播之后，这些缓存就被清空，可通过指定这个参数不清空缓存，用来多次反向传播。
> - create_graph：对反向传播过程再次构建计算图，可通过`backward of backward`实现求高阶导数。

> 计算下面这个函数的导函数：
> $$
> y = x^2\bullet e^x
> $$
> 它的导函数是：
> $$
> {dy \over dx} = 2x\bullet e^x + x^2 \bullet e^x
> $$
> 来看看autograd的计算结果与手动求导计算结果的误差。

```python
import torch as t 

def f(x):
    '''计算y'''
    y = x**2 * t.exp(x)
    return y

def gradf(x):
    '''手动求导函数'''
    dx = 2*x*t.exp(x) + x**2*t.exp(x)
    return dx
  
# 生成数据，由于对x求导，所以设定requires_grad = True
x = t.randn(3,4, requires_grad = True)
y = f(x)
y
# tensor([[ 0.0928,  0.1978,  0.6754,  0.8037],
#         [ 0.9882,  0.3546,  0.2380,  0.0002],
#         [ 0.2863,  0.0448,  0.1516,  2.9122]])
# 此处要注意，对于backward需要传入gradient的尺寸
y.backward(t.ones(y.size())) # gradient形状与y一致
x.grad
# tensor([[ -0.4611,  62.0520,  35.2313,   4.8159],
#         [  1.2937,   2.5127,  -0.2839,  -0.4043],
#         [ -0.3389,   1.9795,   1.6028,   2.0199]])
# 可以看得出来，手动求导的结果和自动求导的结果一致
gradf(x)
# tensor([[ -0.4611,  62.0520,  35.2313,   4.8159],
#         [  1.2937,   2.5127,  -0.2839,  -0.4043],
#         [ -0.3389,   1.9795,   1.6028,   2.0199]])
```

##### `autograd`常用方法

###### `torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None)`

> 计算给定变量计算图的梯度的总和。 该计算图使用链规则进行区分。如果任何Tensor非标量（即它们的数据具有多个元素）并且需要梯度，则该函数另外需要指定grad_tensors。它应该是一个匹配长度的序列，其包含差分函数对应变量的梯度（None对于不需要梯度张量的所有变量，它是可接受的值）。
>
> 此函数在树叶中累加梯度 - 在调用它之前可能需要将其清零。
>
> 参数:
>
> - Tensors(Tensor列表) – 将计算导数的变量 。
> - grad_tensors(序列(`Tensor`或者 `None`)) – 相应张量的每个元素。 对于标量张量或不需要渐变的张量，不能指定任何值。 如果所有grad_tensors都可以接受None值，则此参数是可选的。
> - retain_graph（bool，可选） - 如果为False，则用于计算grad的图形将被释放。请注意，在几乎所有情况下，将此选项设置为True不是必需的，通常可以以更有效的方式解决。默认值为create_graph。
> - create_graph（bool，可选） - 如果为true，则构造导数的图形，允许计算更高阶的衍生产品。默认为False，除非grad_variables包含至少一个非易失性变量。

###### `torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)>`

> 计算并返回输入的输出梯度的总和。
>
> grad*outputs应该是output 包含每个输出的预先计算的梯度的长度匹配序列。如果输出不需要* grad，则梯度可以是None）。当不需要派生图的图形时，梯度可以作为Tensors给出，或者作为Variables，在这种情况下将创建图形。
>
> 如果only_inputs为True，该函数将仅返回指定输入的渐变列表。如果它是False，则仍然计算所有剩余叶子的渐变度，并将累积到其.grad 属性中。
>
> 参数：
>
> - outputs（可变序列） - 差分函数的输出。
> - inputs（可变序列） - 输入将返回梯度的积分（并不积累.grad）。
> - grad_outputs（Tensor 或Variable的序列） - 渐变wrd每个输出。任何张量将被自动转换为volatile，除非create_graph为True。可以为标量变量或不需要grad的值指定无值。如果所有grad_variables都可以接受None值，则该参数是可选的。
> - retain_graph（bool，可选） - 如果为False，则用于计算grad的图形将被释放。请注意，在几乎所有情况下，将此选项设置为True不是必需的，通常可以以更有效的方式解决。默认值为create_graph。
> - create_graph（bool，可选） - 如果为True，则构造导数的图形，允许计算高阶衍生产品。默认为False，除非grad_variables包含至少一个非易失性变量。
> - only_inputs（bool，可选） - 如果为True，则渐变wrt离开是图形的一部分，但不显示inputs不会被计算和累积。默认为True。

**例子**

```python
import torch as t
# 为tensor设置 requires_grad 标识，代表着需要求导数
# pytorch 会自动调用autograd 记录操作
x = t.ones(2, 2, requires_grad=True)

# 上一步等价于
# x = t.ones(2,2)
# x.requires_grad = True
x
# tensor([[ 1.,  1.],
#         [ 1.,  1.]])
y = x.sum()
y
# tensor(4.)
y.grad_fn
# <SumBackward0 at 0x7ffaa589a780>
y.backward() # 反向传播,计算梯度
# y = x.sum() = (x[0][0] + x[0][1] + x[1][0] + x[1][1])
# 每个值的梯度都为1
x.grad 
# tensor([[ 1.,  1.],
#         [ 1.,  1.]])

# 反向传播的时候，梯度注意需要清零
# 未清零，此时的梯度为
y.backward()
x.grad
# tensor([[ 2.,  2.],
#         [ 2.,  2.]])
y.backward()
x.grad
# tensor([[ 3.,  3.],
#         [ 3.,  3.]])
# 清零之后的反向传播
# 以下划线结束的函数是inplace操作，会修改自身的值，就像add_
x.grad.data.zero_()
y.backward()
x.grad
# tensor([[ 1.,  1.],
#         [ 1.,  1.]])
```

> 注意：`grad`在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。

##### 计算图

> `PyTorch`中`autograd`的底层采用了计算图，计算图是一种特殊的有向无环图（`DAG`），用于记录算子与变量之间的关系。一般用矩形表示算子，椭圆形表示变量。如表达式$$ \textbf {z = wx + b}$$可分解为$$\textbf{y = wx}$$和$$\textbf{z = y + b}$$，其计算图如下图所示，图中`MUL`，`ADD`都是算子，$$\textbf{w}$$，$$\textbf{x}$$，$$\textbf{b}$$即变量。
>
> ![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/com_graph.svg)

> 如上有向无环图中，$$\textbf{X}$$和$$\textbf{b}$$是叶子节点（leaf node），这些节点通常由用户自己创建，不依赖于其他变量。$$\textbf{z}$$称为根节点，是计算图的最终目标。利用链式法则很容易求得各个叶子节点的梯度。

$$
\begin{align}
&{\partial z \over \partial b} = 1,\space {\partial z \over \partial y} = 1 \\
&{\partial y \over \partial w }= x,{\partial y \over \partial x}= w \\
&{\partial z \over \partial x}= {\partial z \over \partial y} {\partial y \over \partial x}=1 * w\\
&{\partial z \over \partial w}= {\partial z \over \partial y} {\partial y \over \partial w}=1 * x \\
\end{align}
$$

> 而有了计算图，上述链式求导即可利用计算图的反向传播自动完成。其计算过程如下图所示：
>
> ![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/com_graph_backward.svg)

> 在PyTorch实现中，autograd会随着用户的操作，记录生成当前variable的所有操作，并由此建立一个有向无环图。用户每进行一个操作，相应的计算图就会发生改变。更底层的实现中，图中记录了操作`Function`，每一个变量在图中的位置可通过其`grad_fn`属性在图中的位置推测得到。在反向传播过程中，autograd沿着这个图从当前变量（根节点$$\textbf{z}$$）溯源，可以利用链式求导法则计算所有叶子节点的梯度。每一个前向传播操作的函数都有与之对应的反向传播函数用来计算输入的各个`variable`的梯度，这些函数的函数名通常以`Backward`结尾。

**例子**

```python
import torch as t

# 测试requires_grad
x = t.ones(1)
b = t.rand(1, requires_grad = True)
w = t.rand(1, requires_grad = True)
y = w * x # 等价于y=w.mul(x)
z = y + b # 等价于z=y.add(b)
x.requires_grad, b.requires_grad, w.requires_grad
# (False, True, True)
# 在默认情况下，requires_grad会自动传递
# 虽然未指定y.requires_grad为True，但由于y依赖于需要求导的w
# 故而y.requires_grad为True
y.requires_grad
# True
# 计算图的叶子节点
x.is_leaf, w.is_leaf, b.is_leaf
# (True, True, True)
y.is_leaf, z.is_leaf
# (False, False)
# grad_fn可以查看这个variable的反向传播函数，
# z是add函数的输出，所以它的反向传播函数是AddBackward
z.grad_fn 
# <AddBackward1 at 0x7f60b09c2630>
# next_functions保存grad_fn的输入，是一个tuple，tuple的元素也是Function
# 第一个是y，它是乘法(mul)的输出，所以对应的反向传播函数y.grad_fn是MulBackward
# 第二个是b，它是叶子节点，由用户创建
z.grad_fn.next_functions 
# ((<MulBackward1 at 0x7f60b09c2278>, 0),
# (<AccumulateGrad at 0x7f60b09c2198>, 0))
# variable的grad_fn对应着和图中的function相对应
z.grad_fn.next_functions[0][0] == y.grad_fn

# 第一个是w，叶子节点，需要求导，梯度是累加的
# 第二个是x，叶子节点，不需要求导，所以为None
y.grad_fn.next_functions
# ((<AccumulateGrad at 0x7f60b09c2898>, 0), (None, 0))
# 叶子节点的grad_fn是None
w.grad_fn,x.grad_fn
# (None, None)
```

> 变量的`requires_grad`属性默认为False，如果某一个节点requires_grad被设置为True，那么所有依赖它的节点`requires_grad`都是True。这其实很好理解，对于$$ \textbf{x}\to \textbf{y} \to \textbf{z}$$，x.requires_grad = True，当需要计算$$\partial z \over \partial x$$时，根据链式法则，$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \frac{\partial y}{\partial x}$$，自然也需要求$$ \frac{\partial z}{\partial y}$$，所以y.requires_grad会被自动标为True. 
>
> 有些时候我们可能不希望autograd对tensor求导。认为求导需要缓存许多中间结构，增加额外的内存/显存开销，那么我们可以关闭自动求导。对于不需要反向传播的情景（如inference，即测试推理时），关闭自动求导可实现一定程度的速度提升，并节省约一半显存，因其不需要分配空间计算梯度。

**例子**

```python
# 两种抑制requires_grad传递的方法
x = t.ones(1, requires_grad=True)
w = t.rand(1, requires_grad=True)
y = x * w
# y依赖于w，而w.requires_grad = True
x.requires_grad, w.requires_grad, y.requires_grad
# (True, True, True)
# 1.第一种抑制的方法
with t.no_grad():
    x = t.ones(1)
    w = t.rand(1, requires_grad = True)
    y = x * w
# y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False
x.requires_grad, w.requires_grad, y.requires_grad
# (False, True, False)
# 2.第二种抑制的方法
t.set_grad_enabled(False)
x = t.ones(1)
w = t.rand(1, requires_grad = True)
y = x * w
# y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False
x.requires_grad, w.requires_grad, y.requires_grad
# (False, True, False)
# 恢复默认配置
t.set_grad_enabled(True)
```

> 在反向传播过程中非叶子节点的导数计算完之后即被清空。若想查看这些变量的梯度，有两种方法：
>
> - 使用autograd.grad函数
> - 使用hook
>
> `autograd.grad`和`hook`方法都是很强大的工具，更详细的用法参考官方api文档，这里举例说明基础的使用。推荐使用`hook`方法，但是在实际使用中应尽量避免修改grad的值。

```python
import torch as t 
# 中间节点，梯度自动清零
x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
# y依赖于w，而w.requires_grad = True
z = y.sum()
x.requires_grad, w.requires_grad, y.requires_grad
# (True, True, True)
# 非叶子节点grad计算完之后自动清空，y.grad是None
z.backward()
(x.grad, w.grad, y.grad)
# (tensor([ 0.2709,  0.0473,  0.5052]), tensor([ 1.,  1.,  1.]), None)
# 第一种方法：使用grad获取中间变量的梯度
x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
z = y.sum()
# z对y的梯度，隐式调用backward()
t.autograd.grad(z, y)
# (tensor([ 1.,  1.,  1.]),)

# 第二种方法：使用hook
# hook是一个函数，输入是梯度，不应该有返回值
def variable_hook(grad):
    print('y的梯度：',grad)

x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
# 注册hook
hook_handle = y.register_hook(variable_hook)
z = y.sum()
z.backward()

# 除非你每次都要用hook，否则用完之后记得移除hook
hook_handle.remove()
# y的梯度： tensor([ 1.,  1.,  1.])
```

> 关于variable中grad属性和backward函数`grad_variables`参数的含义：
>
> - variable $$\textbf{x}$$的梯度是目标函数$${f(x)} $$对$$\textbf{x}$$的梯度，$$\frac{df(x)}{dx} = (\frac {df(x)}{dx_0},\frac {df(x)}{dx_1},...,\frac {df(x)}{dx_N})$$，形状和$$\textbf{x}$$一致。
> - 对于y.backward(grad_variables)中的grad_variables相当于链式求导法则中的$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \frac{\partial y}{\partial x}$$中的$$\frac{\partial z}{\partial y}$$。z是目标函数，一般是一个标量，故而$$\frac{\partial z}{\partial y}$$的形状与variable $$\textbf{y}$$的形状一致。`z.backward()`在一定程度上等价于y.backward(grad_y)。`z.backward()`省略了grad_variables参数，是因为$$z$$是一个标量，而$$\frac{\partial z}{\partial z} = 1$$

**例子**

```python
import torch as t 

# 默认参数下的反向传播，默认情况下，z必须是标量
x = t.arange(0,3, requires_grad=True)
y = x**2 + x*2
z = y.sum()
z.backward() # 从z开始反向传播
x.grad
# tensor([ 2.,  4.,  6.])
# 指定Variable的反向传播，如果z不是标量，不指定Variable，会出错
x = t.arange(0,3, requires_grad=True)
y = x**2 + x*2
z = y.sum()
y_gradient = t.Tensor([1,1,1]) # dz/dy
y.backward(y_gradient) #从y开始反向传播
x.grad
# tensor([ 2.,  4.,  6.])
```

> 在`PyTorch`中计算图的特点可总结如下：
>
> - `autograd`根据用户对`variable`的操作构建其计算图。对变量的操作抽象为`Function`。
> - 对于那些不是任何函数(`Function`)的输出，由用户创建的节点称为叶子节点，叶子节点的`grad_fn`为None。叶子节点中需要求导的`variable`，具有`AccumulateGrad`标识，因其梯度是累加的。
> - `variable`默认是不需要求导的，即`requires_grad`属性默认为False，如果某一个节点`requires_grad`被设置为True，那么所有依赖它的节点`requires_grad`都为True。
> - ~~`variable`的`volatile`属性默认为False，如果某一个`variable`的`volatile`属性被设为True，那么所有依赖它的节点`volatile`属性都为True。`volatile`属性为True的节点不会求导，`volatile`的优先级比`requires_grad`高。~~
> - 多次反向传播时，梯度是累加的。反向传播的中间缓存会被清空，为进行多次反向传播需指定`retain_graph=True`来保存这些缓存。
> - 非叶子节点的梯度计算完之后即被清空，可以使用`autograd.grad`或`hook`技术获取非叶子节点的值。
> - `variable`的`grad与data`形状一致，应避免直接修改`variable.data`，因为对`data`的直接操作无法利用autograd进行反向传播
> - 反向传播函数`backward`的参数`grad_variables`可以看成链式求导的中间结果，如果是标量，可以省略，默认为1
> - `PyTorch`采用动态图设计，可以很方便地查看中间层的输出，动态的设计计算图结构。

#### `autograd`高级用法

> `Pytorch`提供的大部分函数能自动实现反向传播，但如果需要自己写一个复杂的函数，不支持自动反向求导的时候，我们就需要手动实现反向传播函数。
>
> `Pytorch`提供了两种方法来扩展`autograd`:
>
> 第一种自定义`Function`：
>
> > - 自定义的Function需要继承`autograd.Function`，没有构造函数`__init__`，`forward`和`backward`函数都是静态方法
> > - `backward`函数的输出和`forward`函数的输入一一对应，`backward`函数的输入和forward函数的输出一一对应
> > - `backward`函数的`grad_output`参数即`t.autograd.backward`中的`grad_variables`
> > - 如果某一个输入不需要求导，直接返回None，如`forward`中的输入参数`x_requires_grad`显然无法对它求导，直接返回None即可
> > - 反向传播可能需要利用前向传播的某些中间结果，需要进行保存，否则前向传播结束后这些对象即被释放
> >
> > `Function`的使用利用`Function.apply(variable)`。
>
> 第二种方法：
>
> > `PyTorch`提供了一个装饰器`@once_differentiable`，能够在`backward`函数中自动将输入的`variable`提取成tensor，把计算结果的tensor自动封装成variable。有了这个特性我们就能够很方便的使用`numpy/scipy`中的函数，操作不再局限于variable所支持的操作。但是这种做法正如名字中所暗示的那样只能求导一次，它打断了反向传播图，不再支持高阶求导。

**例子**

```python
from torch.autograd import Function
class MultiplyAdd(Function):
                                                            
    @staticmethod
    def forward(ctx, w, x, b):                              
        ctx.save_for_backward(w,x)
        output = w * x + b
        return output
        
    @staticmethod
    def backward(ctx, grad_output):                         
        w,x = ctx.saved_tensors
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b             
      
      
x = t.ones(1)
w = t.rand(1, requires_grad = True)
b = t.rand(1, requires_grad = True)
# 开始前向传播
z=MultiplyAdd.apply(w, x, b)
# 开始反向传播
z.backward()
# x不需要求导，中间过程还是会计算它的导数，但随后被清空
x.grad, w.grad, b.grad
# (None, tensor([ 1.]), tensor([ 1.]))
```

### `Pytorch`实例线性回归

> 三种方法实现线性回归，第一种：自动计算导数，第二种：使用`autograd`来计算导数，第三种：使用优化器自动优化

#### 线性回归0

```python
import torch as t
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
from IPython import display
style.use("ggplot")


device = t.device('cuda') #如果你想用gpu，改成t.device('cuda:0')

# 设置随机数种子，保证在不同电脑上运行时下面的输出一致
t.manual_seed(1000) 

def get_fake_data(batch_size=8):
    ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
    x = t.rand(batch_size, 1, device=device) * 5
    y = x * 2 + 3 +  t.randn(batch_size, 1, device=device)
    return x, y 
  
# 随机初始化参数
w = t.rand(1, 1).to(device)
b = t.zeros(1, 1).to(device)

lr =0.02 # 学习率
num_epochs = 500
# 训练500次
for ii in range(num_epochs):
  	# 获取数据
    x, y = get_fake_data(batch_size=4)
    
    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y) # x@W等价于x.mm(w);for python3 only
    loss = 0.5 * (y_pred - y) ** 2 # 均方误差
    loss = loss.mean()
    
    # backward：手动计算梯度
    # 模拟计算图的方式计算误差
    dloss = 1
    dy_pred = dloss * (y_pred - y)
    
    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()
    
    # 更新参数
    w.sub_(lr * dw)
    b.sub_(lr * db)
    
    if ii%50 ==0:
        # 画图
        # 使用display实现动态图
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1).to(device)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.cpu().numpy(), y.cpu().numpy(),c = 'green') # predicted
        
        x2, y2 = get_fake_data(batch_size=32) 
        plt.scatter(x2.cpu().numpy(), y2.cpu().numpy()) # true data
        
        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)
        
print('w: ', w.item(), 'b: ', b.item())
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/huigui.gif)

#### 线性回归1

```python
import torch as t
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
from IPython import display
style.use("ggplot")


# 注意，此处如果使用cuda，那么计算梯度的时候，会出错
device = t.device('cpu') 

# 设置随机数种子，保证在不同电脑上运行时下面的输出一致
t.manual_seed(1000) 

def get_fake_data(batch_size=8):
    ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
    x = t.rand(batch_size, 1, device=device) * 5
    y = x * 2 + 3 +  t.randn(batch_size, 1, device=device)
    return x, y 
  
# 随机初始化参数
w = t.rand(1, 1).to(device)
b = t.zeros(1, 1).to(device)

lr =0.02 # 学习率
num_epochs = 500
# 训练500次

for ii in range(num_epochs):
    x, y = get_fake_data(batch_size=32)
    
    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    losses[ii] = loss.item()
    
    # backward：手动计算梯度
    loss.backward()
    
    # 更新参数
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)
    
    # 梯度清零
    w.grad.data.zero_()
    b.grad.data.zero_()
    
    if ii % 10 ==0:
        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1)
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy(),c = 'blue') # predicted
        print(w.item(),b.item())
        
        x2, y2 = get_fake_data(batch_size=20) 
        plt.scatter(x2.numpy(), y2.numpy()) # true data
        
        plt.xlim(0,5)
        plt.ylim(0,13)   
        plt.show()
        plt.pause(0.5)
        
print(w.item(), b.item())
# 1.9642277956008911 3.0166714191436768
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/huigui_1.gif)

#### 线性回归2

```python
import torch as t
%matplotlib inline
from matplotlib import pyplot as plt
from torch import optim
from torch import nn 
from matplotlib import style
from IPython import display
style.use("ggplot")


# 不知道为啥使用cuda梯度为不存在了
device = t.device('cpu') 

# 设置随机数种子，保证在不同电脑上运行时下面的输出一致
t.manual_seed(1000) 

def get_fake_data(batch_size=8):
    ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
    x = t.rand(batch_size, 1, device=device) * 5
    y = x * 2 + 3 +  t.randn(batch_size, 1, device=device)
    return x, y 
 
# 随机初始化参数
w = t.rand(1,1, requires_grad=True).to(device)
b = t.zeros(1,1, requires_grad=True).to(device)

lr = 0.0005 # 学习率不能太大
num_epochs = 500
losses = np.zeros(500)

# 导入优化器
opt = optim.SGD([w,b],lr = lr)
# 引入mse损失函数
loss_func = nn.MSELoss()

for ii in range(num_epochs):
    x, y = get_fake_data(batch_size=32)
    
    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = loss_func(y_pred,y)
    losses[ii] = loss.item()
    # 优化器梯度清零
    opt.zero_grad()
    # backward：手动计算梯度
    loss.backward()
    # 逐步优化
    opt.step()
    
    
    if ii % 10 == 0:
        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1)
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy(),c = 'blue') # predicted
        print(w.item(),b.item())
        
        x2, y2 = get_fake_data(batch_size=20) 
        plt.scatter(x2.numpy(), y2.numpy()) # true data
        
        plt.xlim(0,5)
        plt.ylim(0,13)   
        plt.show()
        plt.pause(0.5)
        
print(w.item(), b.item())
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/huigui_0.gif)

### 参考

> [深度学习之Pytorch(陈云)](https://github.com/chenyuntc/pytorch-book/tree/master/chapter4-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%B7%A5%E5%85%B7%E7%AE%B1nn)
