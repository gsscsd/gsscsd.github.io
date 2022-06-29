# PyTorch快速入门1


在学习了`PyTorch`的`Tensor、Variable和autograd`之后，已经可以实现简单的深度学习模型，然而使用`autograd`实现的深度学习模型，其抽象程度比较较低，如果用其来实现深度学习模型，则需要编写的代码量极大。在这种情况下，`torch.nn`应运而生，其是专门为深度学习而设计的模块。`torch.nn`的核心数据结构是`Module`，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承`nn.Module`，撰写自己的网络层。

<!--more-->

> - 自定义层`Linear`必须继承`nn.Module`，并且在其构造函数中需调用`nn.Module`的构造函数，即`super(Linear, self).__init__()` 或`nn.Module.__init__(self)`，推荐使用第一种用法，尽管第二种写法更直观。
> - 在构造函数`__init__`中必须自己定义可学习的参数，并封装成`Parameter`，如在本例中我们把`w`和`b`封装成`parameter`。`parameter`是一种特殊的`Tensor`，但其默认需要求导（requires_grad = True），感兴趣的读者可以通过`nn.Parameter??`，查看`Parameter`类的源代码。
> - `forward`函数实现前向传播过程，其输入可以是一个或多个tensor。
> - 无需写反向传播函数，`nn.Module`能够利用`autograd`自动实现反向传播，这点比Function简单许多。
> - 使用时，直观上可将layer看成数学概念中的函数，调用layer(input)即可得到input对应的结果。它等价于`layers.__call__(input)`，在`__call__`函数中，主要调用的是 `layer.forward(x)`，另外还对钩子做了一些处理。所以在实际使用中应尽量使用`layer(x)`而不是使用`layer.forward(x)`。
> - `Module`中的可学习参数可以通过`named_parameters()`或者`parameters()`返回迭代器，前者会给每个parameter都附上名字，使其更具有辨识度。
>
> 可见利用Module实现的全连接层，比利用`Function`实现的更为简单，因其不再需要写反向传播函数。

**例子**

```python
# 用nn.Module实现自己的全连接层。
 # 继承nn.Module
class Linear(nn.Module):
    def __init__(self,input_size,output_size):
        # 父类的初始化函数
        super(Linear, self).__init__()
        # 初始化参数
        self.w = nn.Parameter(t.randn(input_size,output_size))
        self.b = nn.Parameter(t.randn(output_size))
    # 前向传播函数
    def forward(self,x):
        x = x.mm(self.w) 
        # 扩维
        x = x + self.b.expand_as(x)
        return x 
    
# 调用Liner层
layer = Linear(4,3)
input = t.randn(2,4)
output = layer(input)
output
# tensor([[ 0.7737, -0.3466, -1.1029],
#         [-0.9017,  0.1434,  0.1074]])
# 查看网络层的参数
for name, parameter in layer.named_parameters():
    print(name, parameter) # w and b 
"""
w Parameter containing:
tensor([[-2.1490, -0.4335,  0.3410],
        [ 0.4159,  1.4965,  1.6047],
        [ 0.2440, -1.0860,  0.3173],
        [ 1.0884, -0.2328, -0.9765]])
b Parameter containing:
tensor([ 0.0847, -0.4250, -0.2058])
"""
```

**例子**

```python
# 使用上面定义好的Linear模块定义感知机
import torch as t 
from torch import nn
class Perceptron(nn.Module):
  	# 定义输入的参数
    def __init__(self, in_features, hidden_features, out_features):
        nn.Module.__init__(self)
        self.layer1 = Linear(in_features, hidden_features) # 此处的Linear是前面自定义的全连接层
        self.layer2 = Linear(hidden_features, out_features)
    # 前向传播函数
    def forward(self,x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        return self.layer2(x)
      
# 查看网络结构
perceptron = Perceptron(3,4,1)
for name, param in perceptron.named_parameters():
    print(name, param.size())
    
"""
layer1.w torch.Size([3, 4])
layer1.b torch.Size([4])
layer2.w torch.Size([4, 1])
layer2.b torch.Size([1])
"""
```

> module中parameter的命名规范：
>
> - 对于类似`self.param_name = nn.Parameter(t.randn(3, 4))`，命名为`param_name`
> - 对于子Module中的parameter，会其名字之前加上当前Module的名字。如对于`self.sub_module = SubModel()`，SubModel中有个parameter的名字叫做param_name，那么二者拼接而成的parameter name 就是`sub_module.param_name`。

### 常用神经网络层

> 为方便用户使用，PyTorch实现了神经网络中绝大多数的layer，这些layer都继承于nn.Module，封装了可学习参数`parameter`，并实现了forward函数，且很多都专门针对GPU运算进行了CuDNN优化，其速度和性能都十分优异。更多的内容可参照[官方文档](http://pytorch.org/docs/nn.html)或在IPython/Jupyter中使用nn.layer?来查看。阅读文档时应主要关注以下几点：
>
> - 构造函数的参数，如nn.Linear(in_features, out_features, bias)，需关注这三个参数的作用。
> - 属性、可学习参数和子module。如nn.Linear中有`weight`和`bias`两个可学习参数，不包含子module。
> - 输入输出的形状，如nn.linear的输入形状是(N, input_features)，输出为(N，output_features)，N是batch_size。
>
> 这些自定义layer对输入形状都有假设：输入的不是单个数据，而是一个batch。输入只有一个数据，则必须调用`tensor.unsqueeze(0)` 或 `tensor[None]`将数据伪装成batch_size=1的batch

### 图像相关层

> 图像相关层主要包括卷积层（Conv）、池化层（Pool）等，这些层在实际使用中可分为一维(1D)、二维(2D)、三维（3D），池化方式又分为平均池化（AvgPool）、最大值池化（MaxPool）、自适应池化（AdaptiveAvgPool）等。而卷积层除了常用的前向卷积之外，还有逆卷积（TransposeConv）。

> 深度学习当中，经常用到的网络层：
>
> - Linear：全连接层。
> - BatchNorm：批规范化层，分为1D、2D和3D。除了标准的BatchNorm之外，还有在风格迁移中常用到的InstanceNorm层。
> - Dropout：dropout层，用来防止过拟合，同样分为1D、2D和3D。
> - Conv：卷积层，分为1D、2D和3D
> - Pool：池化层，分为1D、2D和3D

```python
from PIL import Image
from torchvision.transforms import ToPILImage,ToTensor
import torch as t 
from torch import nn

to_tensor = ToTensor() # img -> tensor
to_pil = ToPILImage()

img = Image.open("./imgs/lena.png")
img

```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/lena.png)

```python
# 输入是一个batch，batch_size＝1
input = to_tensor(img).unsqueeze(0)
print(input.shape)
# torch.Size([1, 1, 200, 200])
# 锐化卷积核
kernel = t.ones(3, 3)/-9.
kernel[1][1] = 1
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)
out = conv(input)
to_pil(out.data.squeeze(0))
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/lena_conv.png)

```python
pool = nn.AvgPool2d(2,2)
out = pool(input)
to_pil(out.data.squeeze(0))
```

![](https://blog-1253453438.cos.ap-beijing.myqcloud.com/pytorch/lena_pool.png)

### `Sequential`和`ModuleList`

> 使用nn.module实现的神经网络，基本上都是将每一层的输出直接作为下一层的输入，这种网络称为前馈传播网络（`feedforward neural network`）。对于此类网络如果每次都写复杂的forward函数会有些麻烦，在此就有两种简化方式，`ModuleList`和`Sequential`。其中`Sequential`是一个特殊的`module`，它包含几个子`Module`，前向传播时会将输入一层接一层的传递下去。`ModuleList`也是一个特殊的`module`，可以包含几个子`module`，可以像用list一样使用它，但不能直接把输入传给`ModuleList`。下面举例说明。

```python
# 1.Sequential实例
# Sequential的三种写法
net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3, 3, 3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())

net2 = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.BatchNorm2d(3),
        nn.ReLU()
        )
from collections import OrderedDict
net3= nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(3, 3, 3)),
          ('bn1', nn.BatchNorm2d(3)),
          ('relu1', nn.ReLU())
        ]))
print('net1:', net1)
print('net2:', net2)
print('net3:', net3)
"""
net1: Sequential(
  (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  (batchnorm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (activation_layer): ReLU()
)
net2: Sequential(
  (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
)
net3: Sequential(
  (conv1): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
  (bn1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): ReLU()
)
"""
# 可根据名字或序号取出子module
net1.conv, net2[0], net3.conv1
"""
(Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
 Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
 Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)))
 """

# 2.ModuleList实例
modellist = nn.ModuleList([nn.Linear(3,4), nn.ReLU(), nn.Linear(4,2)])
input = t.randn(1, 3)
for model in modellist:
    input = model(input)
    print(input)
"""
tensor([[ 0.2275, -0.3564, -0.0518,  0.5852]])
tensor([[ 0.2275,  0.0000,  0.0000,  0.5852]])
tensor([[-0.4349,  0.3610]])
"""
```

### 激活函数、损失函数与优化器

```python
# 1.激活函数
 nn.ReLU()
# 2.损失函数
nn.CrossEntropyLoss()
# 3.优化器
torch.optim.SGD(params=net.parameters(), lr=1)

##################################################
#### 综合实例
import torch as t 
from torch import nn
from torch import optim

# 首先定义一个LeNet网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
                    nn.Conv2d(3, 6, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(6, 16, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x
# 实例化神经网络
net = Net()
optimizer = optim.SGD(params=net.parameters(), lr=1)
# 第一步
optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()

input = t.randn(1, 3, 32, 32)
output = net(input)
# 第二步
output.backward(output) # fake backward
# 第三步
optimizer.step() # 执行优化
##################################################
# 优化器调参
# 为不同子网络设置不同的学习率，在finetune中经常用到
# 如果对某个参数不指定学习率，就使用最外层的默认学习率
optimizer =optim.SGD([
                {'params': net.features.parameters()}, # 学习率为1e-5
                {'params': net.classifier.parameters(), 'lr': 1e-2}
            ], lr=1e-5)
optimizer
"""
SGD (
Parameter Group 0
    dampening: 0
    lr: 1e-05
    momentum: 0
    nesterov: False
    weight_decay: 0

Parameter Group 1
    dampening: 0
    lr: 0.01
    momentum: 0
    nesterov: False
    weight_decay: 0
)
"""
```

### `nn.Functional`和`nn.Module`

#### `nn.Functional`

> `nn`中还有一个很常用的模块：`nn.functional`，`nn`中的大多数layer，在`functional`中都有一个与之相对应的函数。
>
> `nn.functional`中的函数和`nn.Module`的主要区别在于，用`nn.Module`实现的`layers`是一个特殊的类，都是由`class layer(nn.Module)`定义，会自动提取可学习的参数。而`nn.functional`中的函数更像是纯函数，由`def function(input)`定义。

```python
import torch as t 
from torch import nn


input = t.randn(2, 3)
model = nn.Linear(3, 4)
output1 = model(input)
output2 = nn.functional.linear(input, model.weight, model.bias)
output1 == output2
"""
tensor([[ 1,  1,  1,  1],
        [ 1,  1,  1,  1]], dtype=torch.uint8)
"""
b = nn.functional.relu(input)
b2 = nn.ReLU()(input)
b == b2
"""
tensor([[ 1,  1,  1],
        [ 1,  1,  1]], dtype=torch.uint8)
"""
```

> 在实际应用的时候，如果模型有可学习的参数，最好用`nn.Module`，否则既可以使用`nn.functional`也可以使用`nn.Module`，二者在性能上没有太大差异，具体的使用取决于个人的喜好。
>
> 如激活函数（`ReLU、sigmoid、tanh`），池化（`MaxPool`）等层由于没有可学习参数，则可以使用对应的`functional`函数代替，而对于卷积、全连接等具有可学习参数的网络建议使用`nn.Module`。
>
> 虽然dropout操作也没有可学习操作，但建议还是使用`nn.Dropout`而不是`nn.functional.dropout`，因为dropout在训练和测试两个阶段的行为有所差别，使用`nn.Module`对象能够通过`model.eval`操作加以区分。

```python
import torch as t 
from torch import nn
from torch.nn import functional as F

# 实例1 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.pool(F.relu(self.conv1(x)), 2)
        x = F.pool(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例2
class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(t.randn(3, 4))
        self.bias = nn.Parameter(t.zeros(3))
    def forward(self):
        return F.linear(input, weight, bias)
```

#### `nn.Module`

`nn.Module`基类的构造函数：

```python
def __init__(self):
    self._parameters = OrderedDict()
    self._modules = OrderedDict()
    self._buffers = OrderedDict()
    self._backward_hooks = OrderedDict()
    self._forward_hooks = OrderedDict()
    self.training = True
```

> 其中每个属性的解释如下：
>
> - `_parameters`：字典，保存用户直接设置的parameter，`self.param1 = nn.Parameter(t.randn(3, 3))`会被检测到，在字典中加入一个key为`param1`，value为对应`parameter`的item。而`self.submodule = nn.Linear(3, 4)`中的parameter则不会存于此。
> - `_modules`：子module，通过`self.submodel = nn.Linear(3, 4)`指定的子module会保存于此。
> - `_buffers`：缓存。如`batchnorm`使用momentum机制，每次前向传播需用到上一次前向传播的结果。
> - `_backward_hooks`与`_forward_hooks`：钩子技术，用来提取中间变量，类似variable的hook。
> - `training`：BatchNorm与Dropout层在训练阶段和测试阶段中采取的策略不同，通过判断training值来决定前向传播策略。
>
> 上述几个属性中，`_parameters`、`_modules`和`_buffers`这三个字典中的键值，都可以通过`self.key`方式获得，效果等价于`self._parameters['key']`.

```python
import torch as t 
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 等价与self.register_parameter('param1' ,nn.Parameter(t.randn(3, 3)))
        self.param1 = nn.Parameter(t.rand(3, 3))
        self.submodel1 = nn.Linear(3, 4) 
    def forward(self, input):
        x = self.param1.mm(input)
        x = self.submodel1(x)
        return x
      
net = Net()
net
"""
Net(
  (submodel1): Linear(in_features=3, out_features=4, bias=True)
)
"""
# 1.查看父属性
net._modules
# OrderedDict([('submodel1', Linear(in_features=3, out_features=4, bias=True))])
net._parameters
"""
OrderedDict([('param1', Parameter containing:
              tensor([[ 0.3398,  0.5239,  0.7981],
                      [ 0.7718,  0.0112,  0.8100],
                      [ 0.6397,  0.9743,  0.8300]]))])
"""
# 2.查看子属性param1
net.param1 # 等价于net._parameters['param1']
"""
Parameter containing:
tensor([[ 0.3398,  0.5239,  0.7981],
        [ 0.7718,  0.0112,  0.8100],
        [ 0.6397,  0.9743,  0.8300]])
"""
# 3.查看所有的参数
for name, param in net.named_parameters():
    print(name, param.size())
"""
param1 torch.Size([3, 3])
submodel1.weight torch.Size([4, 3])
submodel1.bias torch.Size([4])
""" 
for name, submodel in net.named_modules():
    print(name, submodel)
"""
 Net(
  (submodel1): Linear(in_features=3, out_features=4, bias=True)
)
submodel1 Linear(in_features=3, out_features=4, bias=True)
"""
```

> `nn.Module`在实际使用中可能层层嵌套，一个module包含若干个子module，每一个子module又包含了更多的子module。为方便用户访问各个子module，nn.Module实现了很多方法，如函数`children`可以查看直接子module，函数`module`可以查看所有的子module（包括当前module）。与之相对应的还有函数`named_childen`和`named_modules`，其能够在返回module列表的同时返回它们的名字。
>
> 对于`batchnorm、dropout、instancenorm`等在训练和测试阶段行为差距巨大的层，如果在测试时不将其training值设为True，则可能会有很大影响，这在实际使用中要千万注意。虽然可通过直接设置`training`属性，来将子module设为`train和eval`模式，但这种方式较为繁琐，因如果一个模型具有多个dropout层，就需要为每个dropout层指定training属性。更为推荐的做法是调用`model.train()`函数，它会将当前module及其子module中的所有training属性都设为True，相应的，`model.eval()`函数会把training属性都设为False。

```python
# 5.查看子module
list(net.named_modules())
"""
[('', Net(
    (submodel1): Linear(in_features=3, out_features=4, bias=True)
  )), ('submodel1', Linear(in_features=3, out_features=4, bias=True))]
"""
# 6.设置train和eval
print(net.training, net.submodel1.training)
net.eval()
net.training, net.submodel1.training
"""
True True
(False, False)
"""
```

> `nn.Module`对象在构造函数中的行为看起来有些怪异，如果想要真正掌握其原理，就需要看两个魔法方法`__getattr__`和`__setattr__`。在Python中有两个常用的`buildin`方法`getattr`和`setattr`，`getattr(obj, 'attr1')`等价于`obj.attr`，如果`getattr`函数无法找到所需属性，Python会转而调用`obj.__getattr__('attr1')`方法，即`getattr`函数无法找到的交给`__getattr__`函数处理，没有实现`__getattr__`或者`__getattr__`也无法处理的就会`raise AttributeError`。`setattr(obj, 'name', value)`等价于`obj.name=value`，如果obj对象实现了`__setattr__`方法，`setattr`会直接调用`obj.__setattr__('name', value)`，否则调用`buildin`方法。总结一下：
>
> - `result  = obj.name`会调用`buildin`函数`getattr(obj, 'name')`，如果该属性找不到，会调用`obj.__getattr__('name')`
> - `obj.name = value`会调用`buildin`函数`setattr(obj, 'name', value)`，如果obj对象实现了`__setattr__`方法，`setattr`会直接调用`obj.__setattr__('name', value')` 
>
> `nn.Module`实现了自定义的`__setattr__`函数，当执行`module.name=value`时，会在`__setattr__`中判断`value`是否为`Parameter`或`nn.Module`对象，如果是则将这些对象加到`_parameters`和`_modules`两个字典中，而如果是其它类型的对象，如`Variable`、`list`、`dict`等，则调用默认的操作，将这个值保存在`__dict__`中。
>
> 因`_modules`和`_parameters`中的`item`未保存在`__dict__`中，所以默认的`getattr`方法无法获取它，因而`nn.Module`实现了自定义的`__getattr__`方法，如果默认的`getattr`无法处理，就调用自定义的`__getattr__`方法，尝试从`_modules`、`_parameters`和`_buffers`这三个字典中获取。

```python
import torch as t 
from torch import nn

module = nn.Module()
module.param = nn.Parameter(t.ones(2, 2))
module._parameters
"""
OrderedDict([('param', Parameter containing:
              tensor([[ 1.,  1.],
                      [ 1.,  1.]]))])
"""

submodule1 = nn.Linear(2, 2)
submodule2 = nn.Linear(2, 2)
module_list =  [submodule1, submodule2]
# 对于list对象，调用buildin函数，保存在__dict__中
module.submodules = module_list
print('_modules: ', module._modules)
print("__dict__['submodules']:",module.__dict__.get('submodules'))
"""
_modules:  OrderedDict()
__dict__['submodules']: [Linear(in_features=2, out_features=2, bias=True), Linear(in_features=2, out_features=2, bias=True)]
"""
module_list = nn.ModuleList(module_list)
module.submodules = module_list
print('ModuleList is instance of nn.Module: ', isinstance(module_list, nn.Module))
print('_modules: ', module._modules)
print("__dict__['submodules']:", module.__dict__.get('submodules'))
"""
ModuleList is instance of nn.Module:  True
_modules:  OrderedDict([('submodules', ModuleList(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
))])
__dict__['submodules']: None
"""

# 即module.param, 会调用module.__getattr__('param')
getattr(module, 'param')
"""
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
"""
```

### `Pytorch`模型保存与加载

> 在`PyTorch`中保存模型十分简单，所有的`Module`对象都具有`state_dict()`函数，返回当前`Module`所有的状态数据。将这些状态数据保存后，下次使用模型时即可利用`model.load_state_dict()`函数将状态加载进来。优化器（optimizer）也有类似的机制，不过一般并不需要保存优化器的运行状态。

```python
# 模型的保存与加载
import torch as t 
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 等价与self.register_parameter('param1' ,nn.Parameter(t.randn(3, 3)))
        self.param1 = nn.Parameter(t.rand(3, 3))
        self.submodel1 = nn.Linear(3, 4) 
    def forward(self, input):
        x = self.param1.mm(input)
        x = self.submodel1(x)
        return x
      
net = Net()

# 保存模型
t.save(net.state_dict(), 'net.pth')

# 加载已保存的模型
net2 = Net()
net2.load_state_dict(t.load('net.pth'))

```

### 在`GPU`上运行

> 将Module放在GPU上运行十分简单，只需两步：
>
> - `model = model.cuda()`：将模型的所有参数转存到GPU
> - `input.cuda()`：将输入数据也放置到GPU上
>
> 至于如何在多个GPU上并行计算，PyTorch也提供了两个函数，可实现简单高效的并行GPU计算
>
> - `nn.parallel.data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None)`
> - `class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)`
>
> 可见二者的参数十分相似，通过`device_ids`参数可以指定在哪些GPU上进行优化，output_device指定输出到哪个GPU上。唯一的不同就在于前者直接利用多GPU并行计算得出结果，而后者则返回一个新的module，能够自动在多GPU上进行并行加速。
>
> `DataParallel`并行的方式，是将输入一个batch的数据均分成多份，分别送到对应的GPU进行计算，各个GPU得到的梯度累加。与Module相关的所有数据也都会以浅复制的方式复制多份，在此需要注意，在module中属性应该是只读的。

```python
# method 1
new_net = nn.DataParallel(net, device_ids=[0, 1])
output = new_net(input)

# method 2
output = nn.parallel.data_parallel(new_net, input, device_ids=[0, 1])
```

### `nn`与`autograd`

> `nn.Module`利用的也是`autograd`技术，其主要工作是实现前向传播。在forward函数中，`nn.Module`对输入的tensor进行的各种操作，本质上都是用到了`autograd`技术。这里需要对比`autograd.Function和nn.Module`之间的区别：
>
> - `autograd.Function`利用了`Tensor`对`autograd`技术的扩展，为`autograd`实现了新的运算op，不仅要实现前向传播还要手动实现反向传播
> - `nn.Module`利用了`autograd`技术，对`nn`的功能进行扩展，实现了深度学习中更多的层。只需实现前向传播功能，`autograd`即会自动实现反向传播
> - `nn.functional`是一些`autograd`操作的集合，是经过封装的函数
>
> 作为两大类扩充`PyTorch`接口的方法，在实际使用中，如果某一个操作，在`autograd`中尚未支持，那么只能实现`Function`接口对应的前向传播和反向传播。如果某些时候利用`autograd`接口比较复杂，则可以利用`Function`将多个操作聚合，实现优化，而如果只是想在深度学习中增加某一层，使用`nn.Module`进行封装则更为简单高效。

### 参考

> [深度学习之Pytorch(陈云)](https://github.com/chenyuntc/pytorch-book/tree/master/chapter4-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%B7%A5%E5%85%B7%E7%AE%B1nn)
