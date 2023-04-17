# keras详细介绍




### 计算图与张量

> 要说Pytorch/Tensorflow/Keras，就不能不提它的符号主义特性
>
> 事实上，Pytorch也好，Tensorflow也好，其实是一款符号主义的计算框架，未必是专为深度学习设计的。假如你有一个与深度学习完全无关的计算任务想运行在GPU上，你完全可以通过Pytorch/Tensorflow编写和运行。

<!--more-->

假如我们要求两个数a和b的和，通常只要把值赋值给a和b，然后计算a+b就可以了，正常人类都是这么写的：

```python
a=3
b=5
z = a + b
```

运行到第一行，a是3，运行到第2行，b是5，然后运行第三行，电脑把a和b的值加起来赋给z了。

计算图的方式：

> a+b这个计算任务，可以分为三步。
>
> - 声明两个变量a，b。建立输出变量z
> - 确立a，b和z的计算关系，z=a+b
> - 将两个数值a和b赋值到变量中，计算结果z
>
> 这种“先确定符号以及符号之间的计算关系，然后才放数据进去计算”的办法，就是符号式编程。当你声明a和b时，它们里面是空的。当你确立z=a+b的计算关系时，a，b和z仍然是空的，只有当你真的把数据放入a和b了，程序才开始做计算。
>
> 符号之间的运算关系，就称为运算图。 
>
> 符号式计算的一大优点是，当确立了输入和输出的计算关系后，在进行运算前我们可以对这种运算关系进行自动化简，从而减少计算量，提高计算速度。另一个优势是，运算图一旦确定，整个计算过程就都清楚了，可以用内存复用的方式减少程序占用的内存。

在Keras，Pytorch和Tensorflow中，参与符号运算的那些变量统一称作张量。张量是矩阵的进一步推广。

> 规模最小的张量是0阶张量，即标量，也就是一个数。
>
> 当我们把一些数有序的排列起来，就形成了1阶张量，也就是一个向量
>
> 如果我们继续把一组向量有序的排列起来，就形成了2阶张量，也就是一个矩阵
>
> 把矩阵摞起来，就是3阶张量，我们可以称为一个立方体，具有3个颜色通道的彩色图片就是一个这样的立方体
>
> 把矩阵摞起来，好吧这次我们真的没有给它起别名了，就叫4阶张量了，不要去试图想像4阶张量是什么样子，它就是个数学上的概念。 

### Keras框架结构

![img](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/keras.png)

> - backend：后端，对Tensorflow和Theano进行封装，完成低层的张量运算、计算图编译等
> - models：模型，模型是层的有序组合，也是层的“容器”，是“神经网络”的整体表示
> - layers：层，神经网络的层本质上规定了一种从输入张量到输出张量的计算规则，显然，整个神经网络的模型也是这样一种张量到张量的计算规则，因此keras的model是layer的子类。

上面的三个模块是Keras最为要紧和核心的三块内容，搭建一个神经网络，就只用上面的内容即可。注意的是，backend虽然很重要，但其内容多而杂，大部分内容都是被其他keras模块调用，而不是被用户直接使用。所以它不是新手马上就应该学的，初学Keras不妨先将backend放一旁，从model和layers学起。

> 为了训练神经网络，必须定义一个神经网络优化的目标和一套参数更新的方式，这部分就是目标函数和优化器：
>
> - objectives：目标函数，规定了神经网络的优化方向
> - optimizers：优化器，规定了神经网络的参数如何更新
>
> 此外，Keras提供了一组模块用来对神经网络进行配置：
>
> - initialization：初始化策略，规定了网络参数的初始化方法
> - regularizers：正则项，提供了一些用于参数正则的方法，以对抗过拟合
> - constraints：约束项，提供了对网络参数进行约束的方法
>
> 为了方便调试、分析和使用网络，处理数据，Keras提供了下面的模块：
>
> - callbacks：回调函数，在网络训练的过程中返回一些预定义/自定义的信息
> - visualization：可视化，用于将网络结构绘制出来，以直观观察
> - preprocessing：提供了一组用于对文本、图像、序列信号进行预处理的函数
> - utils：常用函数库，比较重要的是utils.np_utils中的to_categorical，用于将1D标签转为one-hot的2D标签和convert_kernel函数，用于将卷积核在theano模式和Tensorflow模式之间转换。

### api 详细介绍

#### backend

backend这个模块的主要作用，是对tensorflow和theano的底层张量运算进行了包装。用户不用关心具体执行张量运算的是theano还是tensorflow，就可以编写出能在两个框架下可以无缝对接的程序。backend中的函数要比文档里给出的多得多，完全就是一家百货商店。但一般情况下，文档里给出的那些就已经足够你完成大部分工作了，事实上就连文档里给出的函数大部分情况也不会用，这里提几个比较有用的函数：

> - function：毫无疑问这估计是最有用的一个函数了，function用于将一个计算图（计算关系）编译为具体的函数。典型的使用场景是输出网络的中间层结果。
> - image_ordering和set_image_ordering：这组函数用于返回/设置图片的维度顺序，由于Theano和Tensorflow的图片维度顺序不一样，所以有时候需要获取/指定。典型应用是当希望网络自适应的根据使用的后端调整图片维度顺序时。
> - learning_phase：这个函数的主要作用是返回网络的运行状态，0代表测试，1代表训练。当你需要便携一个在训练和测试是行为不同的层（如Dropout）时，它会很有用。
> - int_shape：这是我最常用的一个函数，用于以整数tuple的形式返回张量的shape。要知道从前网络输出张量的shape是看都看不到的，int_shape可以在debug时起到很大作用。
> - gradients： 求损失函数关于变量的导数，也就是网络的反向计算过程。这个函数在不训练网络而只想用梯度做一点奇怪的事情的时候会很有用，如图像风格转移。

backend的其他大部分函数的函数名是望而知义的，什么max，min，equal，eval，zeros，ones，conv2d等等。函数的命名方式跟numpy差不多，下次想用时不妨先‘.’一下，说不定就有。 

#### models/layers

使用Keras最常见的目的，当然还是训练一个网络。之前说了网络就是张量到张量的映射，所以Keras的网络，其实是一个由多个子计算图构成的大计算图。当这些子计算图是顺序连接时，称为Sequential，否则就是一般的model，我们称为函数式模型。

模型有两套训练和测试的函数，一套是fit，evaluate等，另一套是fit_generator，evaluate_generator，前者适用于普通情况，后者适用于数据是以迭代器动态生成的情况。迭代器可以在内存/显存不足，实时动态数据提升进行网络训练，所以使用Keras的话，Python的迭代器这一部分是一定要掌握的内容。对模型而言，最核心的函数有两个：

> - compile()：编译，模型在训练前必须编译，这个函数用于完成添加正则项啊，确定目标函数啊，确定优化器啊等等一系列模型配置功能。这个函数必须指定的参数是优化器和目标函数，经常还需要指定一个metrics来评价模型。
> - fit()/fit_generator()：用来训练模型，参数较多，是需要重点掌握的函数，对于keras使用者而言，这个函数的每一个参数都需要掌握。

> 另外，模型还有几个常用的属性和函数：
>
> - layers：该属性是模型全部层对象的列表，是的就是一个普通的python list
> - get_layer()：这个函数通过名字来返回模型中某个层对象
> - pop()：这个函数文档里没有，但是可以用。作用是弹出模型的最后一层，从前进行finetune时没有pop，大家一般用model.layers.pop()来完成同样的功能。 
>
> 因为Model是Layer的子类，Layer的所有属性和方法也自动被Model所有

Keras的层对象是构筑模型的基石，除了卷积层，递归神经网络层，全连接层，激活层这种烂大街的Layer对象外，keras还有自己的一套东西：

> - Advanced Activation：高级激活层，主要收录了包括leakyReLU，pReLU，ELU，SReLU等一系列高级激活函数，这些激活函数不是简单的element-wise计算，所以单独拿出来实现一下
> - Merge层：这个层用于将多个层对象的输出组合起来，支持级联、乘法、余弦等多种计算方式，它还有个小兄弟叫merge，这个函数完成与Merge相同的作用，但输入的对象是张量而不是层对象。
> - Lambda层：这是一个神奇的层，看名字就知道它用来把一个函数作用在输入张量上。这个层可以大大减少你的工作量，当你需要定义的新层的计算不是那么复杂的时候，可以通过lambda层来实现，而不用自己完全重写。
> - Highway/Maxout/AtrousConvolution2D层：这个就不多说了，懂的人自然懂，keras还是在一直跟着潮流走的
> - Wrapper层：Wrapper层用于将一个普通的层对象进行包装升级，赋予其更多功能。目前，Wrapper层里有一个TimeDistributed层，用于将普通的层包装为对时间序列输入处理的层，而Bidirectional可以将输入的递归神经网络层包装为双向的（如把LSTM做成BLSTM）
> - Input：补一个特殊的层，Input，这个东西实际上是一个Keras tensor的占位符，主要用于在搭建Model模型时作为输入tensor使用，这个Input可以通过keras.layers来import。
> - stateful与unroll：Keras的递归神经网络层，如SimpleRNN，LSTM等，支持两种特殊的操作。一种是stateful，设置stateful为True意味着训练时每个batch的状态都会被重用于初始化下一个batch的初始状态。另一种是unroll，unroll可以将递归神经网络展开，以空间换取运行时间。

Keras的layers对象还有一些有用的属性和方法:

> - name：别小看这个，从茫茫层海中搜索一个特定的层，如果你对数数没什么信心，最好是name配合get_layer()来用。
> - trainable：这个参数确定了层是可训练的还是不可训练的，在迁移学习中我们经常需要把某些层冻结起来而finetune别的层，冻结这个动作就是通过设置trainable来实现的。
> - input/output：这两个属性是层的输入和输出张量，是Keras tensor的对象，这两个属性在你需要获取中间层输入输出时非常有用
> - get_weights/set_weights：这是两个方法用于手动取出和载入层的参数，set_weights传入的权重必须与get_weights返回的权重具有同样的shape，一般可以用get_weights来看权重shape，用set_weights来载入权重

在Keras中经常有的一个需求是需要自己编写一个新的层，如果你的计算比较简单，那可以尝试通过Lambda层来解决，如果你不得不编写一个自己的层，那也不是什么大不了的事儿。要在Keras中编写一个自己的层，需要开一个从Layer（或其他层）继承的类，除了__init__以为你需要覆盖三个函数：

> - build，这个函数用来确立这个层都有哪些参数，哪些参数是可训练的哪些参数是不可训练的。
> - call，这个函数在调用层对象时自动使用，里面就是该层的计算逻辑，或计算图了。显然，这个层的核心应该是一段符号式的输入张量到输出张量的计算过程。
> - get_output_shape_for：如果你的层计算后，输入张量和输出张量的shape不一致，那么你需要把这个函数也重新写一下，返回输出张量的shape，以保证Keras可以进行shape的自动推断
>
> 由于keras是python编写的，因此，我们可以随时查看keras其他层的源码，参考如何编写

#### 优化器，目标函数，初始化策略

> - objectives是优化目标， 它本质上是一个从张量到数值的函数，当然，是用符号式编程表达的。具体的优化目标有mse，mae，交叉熵等等等等，根据具体任务取用即可，当然，也支持自己编写。需要特别说明的一点是，如果选用categorical_crossentropy作为目标函数，需要将标签转换为one-hot编码的形式，这个动作通过utils.np_utils.to_categorical来完成
> - optimizers是优化器，模型是可以传入优化器对象的，你可以自己配置一个SGD，然后将它传入模型中，参数clipnorm和clipvalue，用来对梯度进行裁剪。
> - activation是激活函数，这部分的内容一般不直接使用，而是通过激活层Activation来调用，此处的激活函数是普通的element-wise激活函数
> - callback是回调函数，这其实是一个比较重要的模块，回调函数不是一个函数而是一个类，用于在训练过程中收集信息或进行某种动作。比如我们经常想画一下每个epoch的训练误差和测试误差，那这些信息就需要在回调函数中收集。预定义的回调函数中**CheckModelpoint，History和EarlyStopping**都是比较重要和常用的。其中CheckPoint用于保存模型，History记录了训练和测试的信息，EarlyStopping用于在已经收敛时提前结束训练。
>
> PS:**History是模型训练函数fit的返回值**

### keras的一些特性

#### 全部Layer都要callable

> Keras的一大性质是**所有的layer对象都是callable的**。所谓callable，就是能当作函数一样来使用，层的这个性质不需要依赖任何模型就能成立。

```python
# 假设我们想要计算x的sigmoid值是多少，我们不去构建model
# 而是构建几个单独的层就可以了
import keras.backend as K
from keras.layers import Activation
import numpy as np

x = K.placeholder(shape=(3,))
y = Activation('sigmoid')(x)
f = K.function([x],[y])
out = f([np.array([1,2,3])])
```

> 把层和模型当作张量的函数来使用，是需要认真贯彻落实的一个东西。
>
> 代码中第1行先定义了一个“占位符”，它的shape是一个长为3的向量。所谓占位符就是“先占个位置 “的符号，翻译成中文就是”此处应有一个长为3的向量“。
>
> 注意第2行，这行我们使用了一个激活层，激活层的激活函数形式是sigmoid，在激活层的后面 又有一个括号，括号内是我们的输入张量x，可以看到，层对象‘Activation('sigmoid')’是被当做一个函数来使用的。层是张量到张量的运算，那么其输出y自然也是一个张量。
>
> 第3行通过调用function函数对计算图进行编译，这个计算图很简单，就是输入张量经过sigmoid作用变成输出向量，计算图的各种优化通过这一步得以完成，现在，f就是一个真正的函数了，就可以按照一般的方法使用了。
>
> **模型也是张量到张量的映射，所以Layer是Model的父类**，因此，一个模型本身也可以像上面一样使用。总而言之，在Keras中，层对象是callable的。

#### Shape与Shape自动推断

> 使用过Keras的都知道，Keras的所有的层有一个“input_shape”的参数，用来指定输入张量的shape。然而这个input_shape，或者有时候是input_dim，只需要在模型的首层加以指定。一旦模型的首层的input_shape指定了，后面的各层就不用再指定，而会根据计算图自动推断。这个功能称为shape的自动推断。
>
> Keras的自动推断依赖于Layer中的get_output_shape_for函数来实现。在所有的Keras中都有这样一个函数，因此后面的层可以通过查看这个函数的返回值获取前层的输入shape，并通过自己的get_output_shape_for将这个信息传递下去。
>
> 然而，有时候，这个自动推断会出错。这种情况发生在一个RNN层后面接Flatten然后又接Dense的时候，这个时候Dense的output_shape无法自动推断出。这时需要指定RNN的输入序列长度input_length，或者在网络的第一层通过input_shape就指定。这种情况极少见，大致有个印象即可，遇到的话知道大概是哪里出了问题就好。
>
> 一般而言，神经网络的数据是以batch为单位的，但在指明input_shape时不需要说明一个batch的样本数。假如你的输入是一个224*224*3的彩色图片，在内部运行时数据的shape是(None，224，224，3)。

#### TH与TF的相爱相杀

现在由于theano已经停止更新，所以keras的默认后端是tensorflow。

> dim_ordering，也就是维度顺序:tf的维度顺序是(224，224，3)，只需要记住这个顺序就行。

####  keras读取模型某一层的输出

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/keras_media.jpg)

```python
# model.layer返回的是一个list，其中每一个元素是model中的层
output = model.layer[3].output

# 查看在layer中的index和名字
layers = model.layers
for i , layer in enumerate(layers):
    print(i, layer)
```

#### keras不易发现的坑

> 当我们做分类任务时：
>
> - output layer的activation="sigmoid"，对应的loss="binary_crossentropy"
>
> -  output layer的activation="softmax"，对应的loss="categorical_crossentropy"
>
> 对于2来说，得到的是**join distribution and a multinomial likelihood**，相当于一个概率分布，和为1；
>
> 对于1来说，得到的是**marginal distribution and a Bernoulli likelihood**, p(y0/x) , p(y1/x) etc。
>
> 如果是multi-label classification，就是一个样本属于多类的情况下，需要用1。否则，如果各个类别之间有相互关系（比如简单的情感分类，如果是正向情感就一定意味着负向情感的概率低），可以使用softmax；如果各个类别之间偏向于独立，可以使用sigmoid。
>
> 对于任务1和2，metric=['acc']的'acc'并不是完全同一个评价方法。我们可以直接令metrics=['binary_accuracy','categorical_accuracy']，在训练过程中会两个结果都输出，这样方便自己的判断。

#### keras中的masking层到底是干什么的

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/mask_keras.jpg)

直接看下面的lstm的例子

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/keras/lstm_keras.jpg)

### 参考链接

> - [Keras使用过程中的tricks和errors(持续更新)](https://zhuanlan.zhihu.com/p/34771270)
> - [一个不负责任的Keras介绍（上）](https://zhuanlan.zhihu.com/p/22129946)
> - [一个不负责任的Keras介绍（中）](https://zhuanlan.zhihu.com/p/22129301)
> - [一个不负责任的Keras介绍（下）](https://zhuanlan.zhihu.com/p/22135796)
