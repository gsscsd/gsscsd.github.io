# tensorflow之Tensor、Variable


### 为什么选择tensorflow

> TensorFlow 无可厚非地能被认定为 神经网络中最好用的库之一。它擅长的任务就是训练深度神经网络.通过使用TensorFlow我们就可以快速的入门神经网络，大大降低了深度学习（也就是深度神经网络）的开发成本和开发难度.。TensorFlow 的开源性，让所有人都能使用并且维护， 巩固它. 使它能迅速更新, 提升。
>
> 现在新版本的tensorflow除了支持Graph Execution之外，还提供了Eager Execution。

<!--more-->

### tensorflow编程思想

> TensorFlow 使用**图**来表示计算任务. 图中的节点被称之为 op (operation 的缩写). 一个 op获得 0 个或多个 Tensor , 执行计算, 产生 0 个或多个 Tensor . 每个 Tensor 是一个类型化的多维数组.tensor也是tensorflow中的核心数据类型。
>
> 一个 TensorFlow 图（graph）描述了计算的过程. 为了进行计算, 图必须在会话（session）里被启动. 会话将图的op分发到诸如 CPU 或 GPU 之类的 设备 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回.
>
> TensorFlow 程序通常被组织成一个**构建阶段**和一个**执行阶段**.
>
> > 在构建阶段, op 的执行步骤被描述成一个图. 
> > 在执行阶段, 使用会话执行执行图中的op.例如,通常在构建阶段创建一个图来表示和训练神经网络,然后在执行阶段反复执行图中的训练 op.

### tensorflow的安装

> Tensorflow 的安装方式很多. 比如官网提供的:
>
> - [Pip 安装](https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip-installation)
> - [Virtualenv 安装](https://www.tensorflow.org/versions/master/get_started/os_setup.html#virtualenv-installation)
> - [Anaconda 安装](https://www.tensorflow.org/versions/master/get_started/os_setup.html#anaconda-installation)
> - [Docker 安装](https://www.tensorflow.org/versions/master/get_started/os_setup.html#docker-installation)
> - [从安装源 安装](https://www.tensorflow.org/versions/master/get_started/os_setup.html#installing-from-sources)

#### pip的安装

```python
pip install tensorflow  #python2
pip3 install tensorflow #python3
pip3 install tensorflow==x.x  # 安装的同时指定版本号
pip3 install tensorflow-gpu #安装tenforflow-gpu版本，注意需要首先配置cuda和cudnn
```

#### conda的安装

```python
conda install tensorflow # 安装cpu版本
conda install tensorflow-gpu # 安装gpu版本，这种方法conda会自动配置cuda和cudnn
```

#### 测试tensorflow

```python
import tensorflow as tf 
```

### tensorflow的基础知识

> - **图（Graph）：**用来表示计算任务，也就我们要做的一些操作。
> - **会话（Session）：**建立会话，此时会生成一张空图；在会话中添加节点和边，形成一张图，一个会话可以有多个图，通过执行这些图得到结果。如果把每个图看做一个车床，那会话就是一个车间，里面有若干个车床，用来把数据生产成结果。
> - **Tensor：**用来表示数据，是我们的原料。
> - **变量（Variable）：**用来记录一些数据和状态，是我们的容器。
> - **注入机制(feed):**通过占位符向模式中传入数据。
> - **取回机制(fetch)**：从模式中取得结果。
>
> 形象的比喻是：把会话看做车间，图看做车床，里面用Tensor做原料，变量做容器，feed和fetch做铲子，把数据加工成我们的结果。

Tensorflow是基于graph的并行计算模型。举个例子，计算`a=(b+c)∗(c+2)`，我们可以将算式拆分成一下：

```python
d = b + c
e = c + 2
a = d * e
```

那么将算式转换成graph后：

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/Simple-graph-example.png)

> 将一个简单的算式搞成这样确实大材小用，但是我们可以通过这个例子发现：`d=b+c`和`e=c+2`是不相关的，也就是可以**并行计算**。对于更复杂的CNN和RNN，graph的并行计算的能力将得到更好的展现。

#### tensorflow的处理结构

> Tensorflow 首先要定义神经网络的结构，然后再把数据放入结构当中去运算和 training。
>
> 因为TensorFlow是采用数据流图（data　flow　graphs）来计算, 所以首先我们得创建一个数据流流图，然后再将我们的数据（数据以张量(tensor)的形式存在）放在数据流图中计算，节点（Nodes）在下图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor)。训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来。

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/TensorFlow-data-flow-graph.gif)

> 如果输入tensor的维度是5000×645000×64，表示有5000个训练样本，每个样本有64个特征，则输入层必须有64个node来接受这些特征。
>
> 上图表示的三层网络包括：输入层(图中的input)、隐藏层(这里取名为ReLU layer表示它的激活函数是ReLU）、输出层(图中的Logit Layer)。
>
> 可以看到，每一层中都有相关tensor流入Gradient节点计算梯度，然后这些梯度tensor进入SGD Trainer节点进行网络优化（也就是update网络参数）。
>
> Tensorflow正是通过graph表示神经网络，实现网络的并行计算，提高效率。下面我们将通过一个简单的例子来介绍TensorFlow的基础语法。

#### tensorflow基础概念之**Tensor(tf.Tensor)**

**Tensor**类是最基本最核心的数据结构了，它表示的是一个操作的输出，但是他并不接收操作输出的值，而是提供了在TensorFlow的Session中计算这些值的方法。 
Tensor类主要有两个目的：

> 1. 一个Tensor能够作为一个输入来传递给其他的操作（Operation），由此构造了一个连接不同操作的数据流，使得TensorFLow能够执行一个表示很大，多步骤计算的图。 
> 2. 在图被“投放”进一个Session中后，Tensor的值能够通过把Tensor传到`Seesion.run（）`这个函数里面去得到结果。相同的，也可以用`t.eval（）`这个函数，其中的t就是你的tensor啦，这个函数可以算是`tf.get_default_session().run(t)`的简便写法。

举例说明：

```python
import tensorflow as tf

#build a graph
print("build a graph")
a=tf.constant([[1,2],[3,4]])
b=tf.constant([[1,1],[0,1]])
print("a:",a)
print("b:",b)
print("type of a:",type(a))
c=tf.matmul(a,b) # 矩阵乘法==np.dot(a,b)
print("c:",c)
print("\n")
#construct a 'Session' to excute the graph
sess=tf.Session()

# Execute the graph and store the value that `c` represents in `result`.
print("excuted in Session")
result_a=sess.run(a)
result_a2=a.eval(session=sess)
print("result_a:\n",result_a)
print("result_a2:\n",result_a2)

result_b=sess.run(b)
print("result_b:\n",result_b)

result_c=sess.run(c)
print("result_c:\n",result_c)
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/tf_tensor.png)

> 整个程序分为3个过程，首先是构建计算图，一开始就用`constant（）`函数生成了两个tensor分别是`a`和`b`（下面有对于constant函数的介绍），然后我们试图直接输出`a`和`b`，但是输出并不是两个矩阵，而是各自度对应的tensor类型。然后我们通过`print("type of a:",type(a))` 这句话来输出a的类型，果然是tensor类型（tensor类）。然后我们把`a`和`b`这两个tensor传递给`tf.matmul（）`函数，这个函数是用来计算矩阵乘法的函数。返回的依然是tensor用`c`来接受。到这里为止，我们可以知道，tensor里面并不负责储存值，想要得到值，得去Session中run。我们可以把这部分看做是创建了一个图但是没有运行这个图。 
>
> 然后我们构造了一个Session的对象用来执行图，`sess=tf.Session()` 。 
> 最后就是在session里面执行之前的东西了，可以把一个tensor传递到`session.run()`里面去，得到其值。等价的也可以用`result_a2=a.eval(session=sess)` 来得到。则返回的结果是numpy.ndarray。

**Tensor类的属性**

> - **device:**表示tensor将被产生的设备名称 
> - **dtype**：tensor元素类型 
> - **graph**：这个tensor被哪个图所有 
> - **name**:这个tensor的名称 
> - **op**：产生这个tensor作为输出的操作（Operation） 
> - **shape**：tensor的形状（返回的是`tf.TensorShape`这个表示tensor形状的类） 
> - **value_index**:表示这个tensor在其操作结果中的索引

**函数：** 

> - **tf.Tensor.consumers()**：返回消耗这个tensor的操作列表
> - **tf.Tensor.eval(feed_dict=None, session=None)**：在一个Seesion里面“评估”tensor的值（其实就是计算),在激发tensor.eval()这个函数之前，tensor的图必须已经投入到session里面，或者一个默认的session是有效的，或者显式指定session。
> - **tf.Tensor.get_shape()**:返回tensor的形状，类型是TensorShape。
> - **tf.Tensor.set_shape(shape)**:设置更新这个tensor的形状。

#### tensorflow基础概念之**Variable（tf.Variable）**

> 通过构造一个**Variable类**的实例在图中添加一个**变量（variable）**
>
> **Variable()**这个构造函数**需要初始值**，这个初始值可以是一个任何类型任何形状的Tensor，**初始值的形状和类型**决定了这个变量的**形状和类型**。构造之后，这个变量的形状和类型就固定了，他的值可以通过assign()函数（或者assign类似的函数）来改变。如果你想要在之后改变变量的形状，你就需要assign()函数同时变量的`validate_shape=False`和任何的Tensor一样，通过**Variable()**创造的变量能够作为图中其他操作的输入使用。你也能够在图中添加节点，通过对变量进行算术操作。

举例说明：

```python
# Variable
import numpy as np
import tensorflow as tf

# 定义变量
w = tf.Variable(initial_value=[[1,2],[3,4]],dtype=tf.float32)
x = tf.Variable(initial_value=[[1,1],[1,1]],dtype=tf.float32)
y = tf.matmul(w,x) # 矩阵乘法==np.dot(w,x)
z = tf.sigmoid(y)
print(z)
# 初始化所有的变量
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    z = session.run(z)
    print(z)
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/tf_variable.png)

**属性：**

> - **device:**这个变量的device 
> - **dtype:**变量的元素类型 
> - **graph:**存放变量的图 
> - **initial_value:**这个变量的初始值 
> - **initializer :**这个变量的初始化器 
> - **name:**这个变脸的名字 
> - **op**：产生这个tensor作为输出的操作（Operation） 

**函数:**

> - **__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)**：创建一个新的变量，初始值为initial_value（这个构造函数会创建两个操作（Op），一个变量OP和一个assignOp来设置变量为其初始化值）
> - **assign(value, use_locking=False)**：为变量指定一个新的值 
> - **assign_add(delta, use_locking=False)**：为这个变量加上一个值 
> - **assign_sub(delta, use_locking=False)**：为这个变量减去一个值 
> - **eval(session=None)**：在一个session里面，计算并且返回这个变量的值。这个不算是构造图的方法，它并不会添加一个操作到图里面。这个便捷方法需要一个包含这个变量的图投放到了一个session里面。要是没有sesssion，那么默认的就会使用默认的session。 
> - **get_shape()**：返回变量的形状
> - **initialized_value()**：返回已经初始化变量的值.你应该使用这个函数来代替使用变量自己来初始化依赖这个变量的值的其他变量。
> - **load(value, session=None)**：把新的值载入到变量里面 
> - **read_value()**：返回这个变量的值，在当前的上下文中读取。返回的是一个含有这个值的Tensor
> - **set_shape(shape)**：改变变量形状 

#### tensorflow基本函数讲解

**tf.constant(value,dtype=None,shape=None,name=’Const’,verify_shape=False)**

> `constant（）`函数应该是出镜率很高的函数之一了，所以这里放在基本函数讲解的第一个。这并不是类，而是一个函数，很多初学者容易误解。
>
> 作用：创建一个常量tensor 
>
> 参数： 
> **value**: 一个dtype类型（如果指定了）的常量值（列表）。要注意的是，要是value是一个列表的话，那么列表的长度不能够超过形状参数指定的大小（如果指定了）。要是列表长度小于指定的，那么多余的由列表的最后一个元素来填充。 
> **dtype**: 返回tensor的类型 
> **shape**: 返回的tensor形状。 
> **name**: tensor的名字 
> **verify_shape**: Boolean that enables verification of a shape of values.

**tf.global_variables()**

> 作用：返回全局变量（global variables）。（全局变量是在分布式环境的机器中共享的变量。）`Variable（）`构造函数或者`get_variable（）`自动地把新的变量添加到 graph collection `GraphKeys.GLOBAL_VARIABLES` 中。这个函数返回这个collection中的内容。

**tf.local_variables()**

> 作用：返回局部变量（local variables）。（局部变量是不做存储用的，仅仅是用来临时记录某些信息的变量。 比如用来记录某些epoch数量等等。） local_variable() 函数会自动的添加新的变量到构造函数或者get_variable（）自动地把新的变量添加到 graph collection `GraphKeys.LOCAL_VARIABLES` 中。这个函数返回这个collection中的内容。

**tf.variables_initializer(var_list, name=’init’)**

> 作用： 返回一个初始化一列变量的操作（Op）。要是你把图“投放进一个”session中后，你就能够通过run 这个操作来初始化变量列表`var_list`中的变量 
> 参数： 
> **var_list**: 带初始化变量列表 
> **name**: 可选，操作的名称。

**tf.global_variables_initializer()** 

> 替代以前老的`tf.initialize_all_variables()` 的新方法。
>
> 作用： 返回一个初始化所有全局变量的操作（Op）。要是你把图“投放进一个”session中后，你就能够通过run 这个操作来初始化所有的全局变量，本质相当于`variable_initializers(global_variables())`

**tf.local_variables_initializer()**

> 作用： 返回一个初始化所有局部变量的操作（Op）。要是你把图“投放进一个”session中后，你就能够通过run 这个操作来初始化所有的局部变量，本质相当于`variable_initializers(local_variables())`

**tf.placeholder(dtype, shape=None, name=None)**

> **作用：** 
> **placeholder**的作用可以理解为**占个位置**，我并不知道这里将会是什么值，但是知道类型和形状等等一些信息，先把这些信息填进去占个位置，然后以后用**feed的方式**来把这些数据“填”进去。返回的就是一个用来用来处理feeding一个值的tensor。 
> 那么feed的时候一般就会在你之后session的run（）方法中用到**feed_dict**这个参数了。这个参数的内容就是你要“喂”给那个placeholder的内容。
>
> **参数：** 
> **dtype:** 将要被fed的元素类型 
> **shape:**（可选） 将要被fed的tensor的形状，要是不指定的话，你能够fed进任何形状的tensor。 
> **name:**（可选）这个操作的名字

#### tensorflow基础实例

##### 常量与图

```python
import tensorflow as tf

#building the graph

'''
创建一个常量操作（op）产生 1x2 矩阵，这个操作（op）作为一个节点添加到默认的图中，但是这里这个矩阵并不是一个值，而是一个tensor。
创建另外一个常量操作产生一个1x2 矩阵（解释如上）
'''
mat1=tf.constant([3.,3.],name="mat1")
mat2=tf.constant([4.,4.],name="mat2")

#matrix sum.
s=tf.add(mat1,mat2)

'''
这个默认的图（grapg）现在已经有3个节点了：两个constan（）操作和一个add（）操作。为了真正的得到这个和的值，你需要把这个图投放到一个session里面执行。
'''

'''
为了得到和的值，我们要运行add 操作（op），因此我们在session里面调用“run（）”函数，把代表add op的输出结果s传到函数里面去。表明我们想从add（）操作得到输出。
'''
with tf.Session() as sess:
    result=sess.run(s)
    print("result:",result)
"""
result: [7.,7.]
"""
```

##### tensor和变量

```python
import tensorflow as tf

#Create a Variable, that will be initialized to the scalar value 0.
state=tf.Variable(0,name="state")
print("the name of this variable:",state.name)

# Create an Op to add 1 to `state`.
one = tf.constant(1)
new_value = tf.add(state, one)
# assign（）的这个函数可以看前面的assign函数的解释，更新变量
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()

# Launch the graph and run the ops.
with tf.Session() as sess:
  # Run the 'init' op
  sess.run(init_op)
  # Print the initial value of 'state'
  print(sess.run(state))
  # Run the op that updates 'state' and print 'state'.
  for _ in range(3):
    sess.run(update)
    print("value of state:",sess.run(state))
 """
 the name of this variable: state:0
 value of state: 1
 value of state: 2
 value of state: 3
 """
```

##### fetches和feeds

> **Fetches**表示一种取的动作，我们有时候需要在操作里面取一些输出，其实就是在执行图的过程中在run（）函数里面传入一个tensor就行，然后就会输出tesnor的结果，比如上面的session.run(state)就可以当做一个fetch的动作啦。当然不仅仅限于fetch一个，你也可以fetch多个tensor。
>
> **feed**我们知道是喂养的意思，这个又怎么理解呢？feed的动作一般和placeholder（）函数一起用，前面说过，placeholder（）起到占位的作用（参考前面的placeholder（）函数），怎么理解呢？假如我有一个（堆）数据，但是我也许只知道他的类型，不知道他的值，我就可以先传进去一个类型，先把这个位置占着。等到以后再把数据“喂”给这个变量。 

```python
import tensorflow as tf

#fetch example
print("#fetch example")
a=tf.constant([1.,2.,3.],name="a")
b=tf.constant([4.,5.,6.],name="b")
c=tf.constant([0.,4.,2.],name="c")
add=a+b
mul=add*c

with tf.Session() as sess:
    result=sess.run([a,b,c,add,mul])
    print("after run:\n",result)

print("\n\n")

#feed example
print("feed example")
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)

with tf.Session() as session:
    result_feed=session.run(output,feed_dict={input1:[2.],input2:[3.]})
    print("result:",result_feed)
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/tensorflow/tf_fetch_feed.png)

### 参考资料

> [莫烦tensorflow教程](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/)


