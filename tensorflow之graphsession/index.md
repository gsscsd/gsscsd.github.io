# tensorflow之Graph、Session


> 学习完tensorflow变量常量等基本量的操作，意味着最基本的东西都有了，使用这些基本的操作，我们就做一些数学运算，至于接下来如何操作基本量和组成更大的计算图，那就需要学习Graph和Session了。

<!--more-->

### tensorflow的基础知识

#### tensorflow基础概念之**Graph(tf.Graph)**

> Graph是一个TensorFlow的一种运算，被表示为一个**数据流的图。**
>
> 一个Graph包含一些操作（Operation）对象，这些对象是**计算节点**。前面说过的Tensor对象，则是表示在不同的操作（operation）间的**数据节点**
>
> 每一个任务都需要一个图，即使不去手动的声明，tensorflow也会在后台默认的构建图，然后将操作添加到图里面。
>
> 一般情况下，我们只需要使用默认生成的图即可，特殊情况下，再去显示的声明多个图。

**属性**：

> - **building_function**:Returns True iff this graph represents a function.
> - **finalized**:返回True，要是这个图被终止了
> - **graph_def_versions**:The GraphDef version information of this graph.
> - **seed**:The graph-level random seed of this graph.
> - **version**:Returns a version number that increases as ops are added to the graph.

**函数**：

> - **add_to_collection(name,value)**：存放值在给定名称的collection里面(因为collection不是sets,所以有可能一个值会添加很多次) .
> - **as_default()**：返回一个上下文管理器,使得这个Graph对象成为当前默认的graph.
> - **finalize()**：结束这个graph,使得他只读(read-only).

#### tensorflow基础概念之**Session(tf.Session)**

> 运行TensorFLow操作（operations）的类,一个Seesion包含了操作对象执行的环境. 
>
> Session是一个比较重要的东西，TensorFlow中只有让Graph（计算图）上的节点Session（会话）中执行，才会得到结果。Session的开启涉及真实的运算，因此比较消耗资源。在使用结束后，务必关闭Session。一般在使用过程中，我们可以通过with上下文管理器来使用Session。

```python
# Using the context manager.
with tf.Session() as sess:
  sess.run(...)
```

**属性**：

> - **graph**：“投放”到session中的图 
> - **graph_def：**图的描述 

**函数**：

> - **tf.Session.__init__(target=”, graph=None, config=None)**：Session构造函数，可以在声明Session的时候指定Graph，如果未指定，则使用默认图。
> - **tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)**：运行操作估算（计算）tensor。
> - **tf.Session.close()**：Session使用之后，一定要关闭
> - **tf.as_default()** ：返回一个上下文管理器，使得这个对象成为当前默认的session/使用with关键字然后可以在with关键字代码块中执行

#### tensorflow之激活函数

> 激活操作提供了在神经网络中使用的不同类型的非线性模型。包括光滑非线性模型(sigmoid, tanh, elu, softplus, and softsign)。连续但是不是处处可微的函数(relu, relu6, crelu and relu_x)。当然还有随机正则化 (dropout) ,所有的激活操作都是作用在每个元素上面的，输出一个tensor和输入的tensor又相同的形状和数据类型。 

**激活函数列表**：

> - **tf.nn.relu** 
> - **tf.nn.relu6** 
> - **tf.nn.crelu** 
> - **tf.nn.elu** 
> - **tf.nn.softplus** 
> - **tf.nn.softsign** 
> - **tf.nn.dropout** 
> - **tf.nn.bias_add** 
> - **tf.sigmoid** 
> - **tf.tanh**
> - **tf.nn.softmax**

**tf.nn.relu(features, name=None)**：

> 计算修正线性单元:`max(features,0)`
>
> 如果，你不知道使用哪个激活函数，那么使用relu准没错。

**tf.nn.softmax(logits,dim=-1,name=None)**：

> 计算softmax激活值，多分类输出层的激活函数。

#### tensorflow之优化器

> 深度学习常见的是对于梯度的优化，也就是说，优化器最后其实就是各种对于梯度下降算法的优化，此处记录一下tensorflow的优化器api。

**优化器列表**：

> - **tf.train.Optimizer** 
> - **tf.train.GradientDescentOptimizer** 
> - **tf.train.AdagradOptimizer** 
> - **tf.train.AdadeltaOptimizer** 
> - **tf.train.MomentumOptimizer** 
> - **tf.train.AdamOptimizer** 
> - **tf.train.FtrlOptimizer** 
> - **tf.train.RMSPropOptimizer**

**tf.train.Optimizer**：

> 优化器（optimizers）类的基类。这个类定义了在训练模型的时候添加一个操作的API。**你基本上不会直接使用这个类**。

**tf.train.GradientDescentOptimizer(learning_rate,use_locking=False,name='GradientDescent')** ：

> 作用：创建一个梯度下降优化器对象 
> 参数： 
> **learning_rate:** A Tensor or a floating point value. 要使用的学习率 
> **use_locking:** 要是True的话，就对于更新操作（update operations.）使用锁 
> **name:** 名字，可选，默认是”GradientDescent”.

**tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')** 

> 作用：构造一个使用Adadelta算法的优化器 
> 参数： 
> **learning_rate:** tensor或者浮点数，学习率 
> **rho:** tensor或者浮点数. 优化参数
> **epsilon:** tensor或者浮点数. 优化参数
> **use_locking**: If True use locks for update operations. 
> **name:** 【可选】这个操作的名字，默认是”Adadelta”

#### tensorflow变量作用域机制

> 在深度学习中，我们可能需要用到大量的变量集，而且这些变量集可能在多处都要用到。例如，训练模型时，训练参数如权重（weights）、偏置（biases）等已经定下来，要拿到验证集去验证，我们自然希望这些参数是同一组。以往写简单的程序，可能使用全局限量就可以了，但在深度学习中，这显然是不行的，一方面不便管理，另外这样一来代码的封装性受到极大影响。因此，TensorFlow提供了一种变量管理方法：变量作用域机制，以此解决上面出现的问题。
>
> 在Tensoflow中，提供了两种作用域：
>
> - 命名域(name scope)：通过tf.name_scope()来实现；
> - 变量域（variable scope）：通过tf.variable_scope()来实现；可以通过设置reuse 标志以及初始化方式来影响域下的变量。 
>
> 这两种作用域都会给tf.Variable()创建的变量加上词头，而tf.name_scope对tf.get_variable()创建的变量没有词头影响。

#####  tf.name_scope(‘scope_name’)

> tf.name_scope 主要结合 tf.Variable() 来使用，方便参数命名管理。

```python
import tensorflow as tf

# 与 tf.Variable() 结合使用。简化了命名
with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

# 注意，这里的 with 和 python 中其他的 with 是不一样的
# 执行完 with 里边的语句之后，这个 conv1/ 和 conv2/ 空间还是在内存中的。这时候如果再次执行上面的代码
# 就会再生成其他命名空间
# 下面是在另外一个命名空间来定义变量的
with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

# 所以，实际上weights1 和 weights2 这两个引用名指向了不同的空间，不会冲突
print(weights1.name)
print(weights2.name)
# -----------------
# conv1/weights:0
# conv2/weights:0
```

#####  tf.variable_scope(‘scope_name’)

> tf.variable_scope() 主要结合 tf.get_variable() 来使用，实现 变量共享。

```python
# 这里是正确的打开方式
# 可以看出，name 参数才是对象的唯一标识
import tensorflow as tf
with tf.variable_scope('v_scope') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2,3])
    bias1 = tf.get_variable('bias', shape=[3])

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')
    # 也可以与tf.Variable一起使用
    bias2 = tf.Variable([0.52], name='bias')

print(Weights1.name)
print(Weights2.name)
print(bias2.name)
# 可以看到这两个引用名称指向的是同一个内存对象
# --------------
# v_scope/Weights:0
# v_scope/Weights:0
# v_scope_1/bias:0
```

> tf.variable_scope(‘scope_name’)中的参数reuse很重要，reuse标记变量是否进行复用。
>
> reuse的参数：
>
> - None:默认参数，此时基础父scope的reuse标记
> - tf.AUTO_REUSE:自动复用，如果变量存在，则复用，不存在则创建。
> - True:复用
