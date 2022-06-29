# python之Pandas笔记


> Pandas 是基于 NumPy 构建的库，在数据处理方面可以把它理解为 NumPy 加强版，同时 Pandas 也是一项开源项目。它基于 Cython，因此读取与处理数据非常快，并且还能轻松处理浮点数据中的缺失数据（表示为 NaN）以及非浮点数据。
>
> pandas适合于许多不同类型的数据，包括：
>
> - 具有异构类型列的表格数据，例如SQL表格或Excel数据
> - 有序和无序（不一定是固定频率）时间序列数据。
> - 具有行列标签的任意矩阵数据（均匀类型或不同类型）
> - 任何其他形式的观测/统计数据集。

<!--more-->

### **核心数据结构**

> pandas最核心的就是Series和DataFrame两个数据结构。
>
> DataFrame可以看做是Series的容器，即：一个DataFrame中可以包含若干个Series。

| 名称        | 维度   | 说明                         |
| --------- | ---- | -------------------------- |
| Series    | 1维   | 带有标签的同构类型数组                |
| DataFrame | 2维   | 表格结构，带有标签，大小可变，且可以包含异构的数据列 |

#### Series

##### Series的创建

```python
pandas.Series( data, index, dtype, copy)。
```

| 编号   | 参数      | 描述                                       |
| ---- | ------- | ---------------------------------------- |
| 1    | `data`  | 数据采取各种形式，如：`ndarray`，`list`，`constants`  |
| 2    | `index` | 索引值必须是唯一的和散列的，与数据的长度相同。 默认`np.arange(n)`如果没有索引被传递。 |
| 3    | `dtype` | `dtype`用于数据类型。如果没有，将推断数据类型               |
| 4    | `copy`  | 复制数据，默认为`false`。                         |

```python
# series的创建
# 1.根据list创建
import pandas as pd 
obj = pd.Series([4,7,-5,3])
obj
# 输出
"""
0    4
1    7
2   -5
3    3
dtype: int64
"""
obj.index
#RangeIndex(start=0, stop=4, step=1)
obj.values
#array([ 4,  7, -5,  3])
# 2.根据dict创建
sdata = {'Ohio':35000,'Texas':71000,'Oregon':16000,'Utah':5000}
obj3 = pd.Series(sdata)
obj3 
# 输出
"""
Ohio      35000
Oregon    16000
Texas     71000
Utah       5000
dtype: int64
"""
```

##### Series基本功能

| 编号   | 属性或方法    | 描述                  |
| ---- | -------- | ------------------- |
| 1    | `axes`   | 返回行轴标签列表。           |
| 2    | `dtype`  | 返回对象的数据类型(`dtype`)。 |
| 3    | `empty`  | 如果系列为空，则返回`True`。   |
| 4    | `ndim`   | 返回底层数据的维数，默认定义：`1`。 |
| 5    | `size`   | 返回基础数据中的元素数。        |
| 6    | `values` | 将系列作为`ndarray`返回。   |
| 7    | `head()` | 返回前`n`行。            |
| 8    | `tail()` | 返回最后`n`行。           |

```python
# Series基本功能代码
import pandas as pd
import numpy as np

#Create a series with 100 random numbers
s = pd.Series(np.random.randn(4))
# 1.显示axes
s.axes
"""
The axes are:
[RangeIndex(start=0, stop=4, step=1)]
"""
# 2.显示empty
s.empty
"""
False
"""
# 3.显示ndim
s.ndim
"""
1
"""
# 4.显示size
s.size
"""
4
"""
# 5.values实例
s.values
"""
[ 1.78737302 -0.60515881 0.18047664 -0.1409218 ]
"""
# 6.head默认显示前5个数据
# 7.tail默认显示后5个数据
```

##### Series其他方法

```python
# 1.索引
obj2[2]
#-5
obj2['a']
#-5
obj2[['a','b','d']]
#输出
a   -5
b    7
d    4
dtype: int64
  
# 2.切片
# 与利用下标进行切片不同，使用标签进行切片时，末端是包含的
obj['b':'c']
# 输出
b    1.0
c    2.0
dtype: float64
# 3.重建索引
obj2 = pd.Series([4,7,-5,3],index=['d','b','a','c'])
obj3 = obj2.reindex(['a','b','c','d','e'])
obj3
# 输出
a   -5.0
b    7.0
c    3.0
d    4.0
e    NaN
dtype: NaN,
# reindex出现NaN，使用fill_value属性对NaN填充
obj4 = obj2.reindex(['a','b','c','d','e'],fill_value=0)
obj4
#输出
a   -5
b    7
c    3
d    4
e    0
dtype: int64
# 4.数据运算
# 可以对Series进行numpy中的一些数组运算
np.exp(obj2)
# Series在算术运算中会自动对齐不同索引的数据：
obj3 + obj4
#输出
California         NaN
Ohio           70000.0
Oregon         32000.0
Texas         142000.0
Utah               NaN
dtype: float64
# 5.排序和排名
# sort_index根据索引排序
# sort_values根据值排序
obj = pd.Series(range(4),index=['d','a','b','c'])
obj.sort_index()
#输出：
a    1
b    2
c    3
d    0
dtype: int64
obj.sort_values()
#输出：
d    0
a    1
b    2
c    3
dtype: int64
# 使用rank函数会增加一个排名值
obj = pd.Series([7,-5,7,4,2,0,4])
obj.rank()
#输出：
0    6.5
1    1.0
2    6.5
3    4.5
4    3.0
5    2.0
6    4.5
dtype: float64

obj.rank(method='first')
#输出
0    6.0
1    1.0
2    7.0
3    4.0
4    3.0
5    2.0
6    5.0
dtype: float64
# 6.汇总和计算描述统计
# sum、mean、max、count、median等等
# corr相关系数
# conv协方差
obj1 = pd.Series(np.arange(10),index = list('abcdefghij'))
obj2 = pd.Series(np.arange(12),index = list('cdefghijklmn'))
obj1.corr(obj2)
#1.0
obj1.cov(obj2)
#6.0
# 7.唯一数、值计数
obj = pd.Series(['c','a','d','a','a','b','b','c','c'])
uniques = obj.unique()
uniques
#array(['c', 'a', 'd', 'b'], dtype=object)
#value_counts()返回各数的计数
obj.value_counts()
#输出
a    3
c    3
b    2
d    1
dtype: int64
# 8.处理缺失数据
# isnull判断NaN值
# fillna填充NaN值
# dropna舍弃NaN值
data = pd.Series([1,np.nan,3.5,np.nan,7])
data.fillna(0)
#输出
0    1.0
1    0.0
2    3.5
3    0.0
4    7.0
dtype: float64
```

#### DataFrame

##### DataFrame创建

```python
pandas.DataFrame( data, index, columns, dtype, copy)
```

| 编号   | 参数        | 描述                                       |
| ---- | --------- | ---------------------------------------- |
| 1    | `data`    | 数据采取各种形式，如:`ndarray`，`series`，`map`，`lists`，`dict`，`constant`和另一个`DataFrame`。 |
| 2    | `index`   | 对于行标签，要用于结果帧的索引是可选缺省值`np.arrange(n)`，如果没有传递索引值。 |
| 3    | `columns` | 对于列标签，可选的默认语法是 - `np.arange(n)`。 这只有在没有索引传递的情况下才是这样。 |
| 4    | `dtype`   | 每列的数据类型。                                 |
| 5    | `copy`    | 如果默认值为`False`，则此命令(或任何它)用于复制数据。          |

```python
# DataFrame创建
# 1.使用list创建
import pandas as pd
data = [1,2,3,4,5]
df = pd.DataFrame(data)
df
"""
     0
0    1
1    2
2    3
3    4
4    5
"""
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
df
"""
      Name      Age
0     Alex      10
1     Bob       12
2     Clarke    13
"""
# 2.使用dict创建
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
df
"""
      Age      Name
0     28        Tom
1     34       Jack
2     29      Steve
3     42      Ricky
"""
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df
"""
      one    two
a     1.0    1
b     2.0    2
c     3.0    3
d     NaN    4
"""
```

##### DataFrame基本功能

| 编号   | 属性或方法    | 描述                                       |
| ---- | -------- | ---------------------------------------- |
| 1    | `T`      | 转置行和列。                                   |
| 2    | `axes`   | 返回一个列，行轴标签和列轴标签作为唯一的成员。                  |
| 3    | `dtypes` | 返回此对象中的数据类型(`dtypes`)。                   |
| 4    | `empty`  | 如果`NDFrame`完全为空[无项目]，则返回为`True`; 如果任何轴的长度为`0`。 |
| 5    | `ndim`   | 轴/数组维度大小。                                |
| 6    | `shape`  | 返回表示`DataFrame`的维度的元组。                   |
| 7    | `size`   | `NDFrame`中的元素数。                          |
| 8    | `values` | NDFrame的Numpy表示。                         |
| 9    | `head()` | 返回开头前`n`行。                               |
| 10   | `tail()` | 返回最后`n`行。                                |

```python
# DataFrame例子
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Minsu','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

#Create a DataFrame
df = pd.DataFrame(d)
df
"""
    Age   Name    Rating
0   25    Tom     4.23
1   26    James   3.24
2   25    Ricky   3.98
3   23    Vin     2.56
4   30    Steve   3.20
5   29    Minsu   4.60
6   23    Jack    3.80
"""
# 1.转置
df.T
"""
         0     1       2      3      4      5       6
Age      25    26      25     23     30     29      23
Name     Tom   James   Ricky  Vin    Steve  Minsu   Jack
Rating   4.23  3.24    3.98   2.56   3.2    4.6     3.8
"""
# 2.axes
df.axes
"""
[RangeIndex(start=0, stop=7, step=1), Index([u'Age', u'Name', u'Rating'],
dtype='object')]
"""
# 3.数据类型
df.dtypes
"""
Age     int64
Name    object
Rating  float64
dtype: object
"""
# 4.shape
df.shape
#(7,3)
# 5.values
df.values
"""
[[25 'Tom' 4.23]
[26 'James' 3.24]
[25 'Ricky' 3.98]
[23 'Vin' 2.56]
[30 'Steve' 3.2]
[29 'Minsu' 4.6]
[23 'Jack' 3.8]]
"""
```

### **Pandas常用函数**

#### 统计函数

| 编号   | 函数              | 描述       |
| ---- | --------------- | -------- |
| 1    | `count()`       | 非空观测数量   |
| 2    | `sum()`         | 所有值之和    |
| 3    | `mean()`        | 所有值的平均值  |
| 4    | `median()`      | 所有值的中位数  |
| 5    | `mode()`        | 值的模值     |
| 6    | `std()`         | 值的标准偏差   |
| 7    | `min()`         | 所有值中的最小值 |
| 8    | `max()`         | 所有值中的最大值 |
| 9    | `abs()`         | 绝对值      |
| 10   | `prod()`        | 数组元素的乘积  |
| 11   | `cumsum()`      | 累计总和     |
| 12   | `cumprod()`     | 累计乘积     |
| 13   | ` pct_change()` | 计算变化百分比  |
| 14   | `conv()`        | 计算协方差    |
| 15   | `corr()`        | 相关性计算    |
| 16   | `rank()`        | 数据排名     |

```python
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Minsu','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}

#Create a DataFrame
df = pd.DataFrame(d)
df
"""
 	Age  Name   Rating
0   25   Tom     4.23
1   26   James   3.24
2   25   Ricky   3.98
3   23   Vin     2.56
4   30   Steve   3.20
5   29   Minsu   4.60
6   23   Jack    3.80
7   34   Lee     3.78
8   40   David   2.98
9   30   Gasper  4.80
10  51   Betina  4.10
11  46   Andres  3.65
"""
# 1.sum(),默认axis=0
df.sum()
"""
Age                                                    382
Name     TomJamesRickyVinSteveMinsuJackLeeDavidGasperBe...
Rating                                               44.92
dtype: object
"""
# 2.mean()
df.mean()
"""
Age       31.833333
Rating     3.743333
dtype: float64
"""
# 3.std()
df.std()
"""
Age       9.232682
Rating    0.661628
dtype: float64
"""

# 汇总函数 describe()
df.describe()
"""
               Age         Rating
count    12.000000      12.000000
mean     31.833333       3.743333
std       9.232682       0.661628
min      23.000000       2.560000
25%      25.000000       3.230000
50%      29.500000       3.790000
75%      35.500000       4.132500
max      51.000000       4.800000
"""
```

#### 应用函数

> 要将自定义或其他库的函数应用于*Pandas*对象，有三个重要的方法，下面来讨论如何使用这些方法。使用适当的方法取决于函数是否期望在整个`DataFrame`，行或列或元素上进行操作。
>
> - 表合理函数应用：`pipe()`
> - 行或列函数应用：`apply()`
> - 元素函数应用：`applymap()`

```python
# 1.表格函数应用
import pandas as pd
import numpy as np

def adder(ele1,ele2):
   return ele1+ele2

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.pipe(adder,2)
"""
        col1       col2       col3
0   2.176704   2.219691   1.509360
1   2.222378   2.422167   3.953921
2   2.241096   1.135424   2.696432
3   2.355763   0.376672   1.182570
4   2.308743   2.714767   2.130288
"""
# 2.行列应用函数
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.apply(np.mean)
"""
     col1       col2        col3                                                      
0   0.343569  -1.013287    1.131245 
1   0.508922  -0.949778   -1.600569 
2  -1.182331  -0.420703   -1.725400
3   0.860265   2.069038   -0.537648
4   0.876758  -0.238051    0.473992
"""

# 3.元素函数应用
import pandas as pd
import numpy as np

# My custom function
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.applymap(lambda x:x*100)
"""
         col1         col2         col3
0   17.670426    21.969052    -49.064031
1   22.237846    42.216693     195.392124
2   24.109576   -86.457646     69.643171
3   35.576312   -162.332803   -81.743023
4   30.874333    71.476717     13.028751
"""
```

#### 迭代函数

> `Pandas`对象之间的基本迭代的行为取决于类型。当迭代一个系列时，它被视为数组式，基本迭代产生这些值。其他数据结构，如：`DataFrame`，遵循类似惯例迭代对象的键。
>
> 要遍历数据帧(`DataFrame`)中的行，可以使用以下函数 -
>
> - `iteritems()` - 迭代`(key，value)`对
> - `iterrows()` - 将行迭代为(索引，系列)对
> - `itertuples()` - 以`namedtuples`的形式迭代行

```python
# 1.iteritems,迭代的计算列
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])
for key,value in df.iteritems():
   print (key,value)
"""
col1 0    0.802390
1    0.324060
2    0.256811
3    0.839186
Name: col1, dtype: float64

col2 0    1.624313
1   -1.033582
2    1.796663
3    1.856277
Name: col2, dtype: float64

col3 0   -0.022142
1   -0.230820
2    1.160691
3   -0.830279
Name: col3, dtype: float64
"""
# 2.iterrows 迭代计算行
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])
for row_index,row in df.iterrows():
   print (row_index,row)
    
"""
0  col1    1.529759
   col2    0.762811
   col3   -0.634691
Name: 0, dtype: float64

1  col1   -0.944087
   col2    1.420919
   col3   -0.507895
Name: 1, dtype: float64

2  col1   -0.077287
   col2   -0.858556
   col3   -0.663385
Name: 2, dtype: float64
3  col1    -1.638578
   col2     0.059866
   col3     0.493482
Name: 3, dtype: float64
"""
```

#### 字符与文本处理函数

| 编号   | 函数                    | 描述                                       |
| ---- | --------------------- | ---------------------------------------- |
| 1    | `lower()`             | 将`Series/Index`中的字符串转换为小写。               |
| 2    | `upper()`             | 将`Series/Index`中的字符串转换为大写。               |
| 3    | `len()`               | 计算字符串长度。                                 |
| 4    | `strip()`             | 帮助从两侧的系列/索引中的每个字符串中删除空格(包括换行符)。          |
| 5    | `split(' ')`          | 用给定的模式拆分每个字符串。                           |
| 6    | `cat(sep=' ')`        | 使用给定的分隔符连接系列/索引元素。                       |
| 7    | `get_dummies()`       | 返回具有单热编码值的数据帧(DataFrame)。                |
| 8    | `contains(pattern)`   | 如果元素中包含子字符串，则返回每个元素的布尔值`True`，否则为`False`。 |
| 9    | `replace(a,b)`        | 将值`a`替换为值`b`。                            |
| 10   | `repeat(value)`       | 重复每个元素指定的次数。                             |
| 11   | `count(pattern)`      | 返回模式中每个元素的出现总数。                          |
| 12   | `startswith(pattern)` | 如果系列/索引中的元素以模式开始，则返回`true`。              |
| 13   | `endswith(pattern)`   | 如果系列/索引中的元素以模式结束，则返回`true`。              |
| 14   | `find(pattern)`       | 返回模式第一次出现的位置。                            |
| 15   | `findall(pattern)`    | 返回模式的所有出现的列表。                            |
| 16   | `swapcase`            | 变换字母大小写。                                 |
| 17   | `islower()`           | 检查系列/索引中每个字符串中的所有字符是否小写，返回布尔值            |
| 18   | `isupper()`           | 检查系列/索引中每个字符串中的所有字符是否大写，返回布尔值            |
| 19   | `isnumeric()`         | 检查系列/索引中每个字符串中的所有字符是否为数字，返回布尔值。          |

> 几乎这些方法都使用[Python字符串函数]( http://docs.python.org/3/library/stdtypes.html#string-methods)。 因此，将Series对象转换为String对象，然后执行以上操作即可。

#### 窗口函数

> - `rolling()`：滚动函数
> - `expanding()`：展开函数
> - `ewm()`：指数移动窗口
> - `shift()`：移位函数
>
> 窗口函数主要搭配统计函数来计算。

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10, 4),
index = pd.date_range('1/1/2020', periods=10),
columns = ['A', 'B', 'C', 'D'])

# 1.rolling函数
# 由于窗口大小为3(window)，前两个元素有空值，第三个元素的值将是n，n-1和n-2元素的平均值
df.rolling(window=3).mean()
"""
                  A         B         C         D
2020-01-01       NaN       NaN       NaN       NaN
2020-01-02       NaN       NaN       NaN       NaN
2020-01-03 -0.306293  0.214001 -0.076004 -0.200793
2020-01-04  0.236632 -0.437033  0.046111 -0.252062
2020-01-05  0.761818 -0.181635 -0.546929 -0.738482
2020-01-06  1.306498 -0.411834 -0.680948 -0.070285
2020-01-07  0.956877 -0.749315 -0.503484  0.160620
2020-01-08  0.354319 -1.067165 -1.238036  1.051048
2020-01-09  0.262081 -0.898373 -1.059351  0.342291
2020-01-10  0.326801 -0.350519 -1.064437  0.749869
"""

# 2.expading函数
df.expanding(min_periods=3).mean()
"""
                  A         B         C         D
2018-01-01       NaN       NaN       NaN       NaN
2018-01-02       NaN       NaN       NaN       NaN
2018-01-03 -0.425085 -0.124270 -0.324134 -0.234001
2018-01-04 -0.293824 -0.038188 -0.172855  0.447226
2018-01-05 -0.516146 -0.013441 -0.384935  0.379267
2018-01-06 -0.614905  0.290308 -0.594635  0.414396
2018-01-07 -0.606090  0.121265 -0.604148  0.246296
2018-01-08 -0.597291  0.075374 -0.425182  0.092831
2018-01-09 -0.380505  0.074956 -0.253081  0.146426
2018-01-10 -0.235030  0.018936 -0.259566  0.315200
"""
# 3.ewm函数
df.ewm(com=0.5).mean()
"""
                   A         B         C         D
2019-01-01  1.047165  0.777385 -1.286948 -0.080564
2019-01-02  0.484093 -0.630998 -0.975172 -0.117832
2019-01-03  0.056189  0.830492  0.116325  1.005547
2019-01-04 -0.363824  1.222173  0.497901 -0.235209
2019-01-05 -0.260685  1.066029  0.391480  1.196190
2019-01-06  0.389649  1.458152 -0.231936 -0.481003
2019-01-07  1.071035 -0.016003  0.387420 -0.170811
2019-01-08 -0.573686  1.052081  1.218439  0.829366
2019-01-09  0.222927  0.556430  0.811838 -0.562096
2019-01-10  0.224624 -1.225446  0.204961 -0.800444
"""
```



### Pandas索引与选择

> `Python`和`NumPy`索引运算符`"[]"`和属性运算符`"."`。 可以在广泛的用例中快速轻松地访问*Pandas*数据结构。然而，由于要访问的数据类型不是预先知道的，所以直接使用标准运算符具有一些优化限制。

*`Pandas`*现在支持三种类型的多轴索引：

| 编号   | 索引        | 描述      |
| ---- | --------- | ------- |
| 1    | `.loc()`  | 基于标签    |
| 2    | `.iloc()` | 基于整数    |
| 3    | `.ix()`   | 基于标签和整数 |

```python
# 1 .loc()
# loc需要两个单/列表/范围运算符，用","分隔。第一个表示行，第二个表示列。
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(8, 4),
index = ['a','b','c','d','e','f','g','h'], columns = ['A', 'B', 'C', 'D'])
df.loc[:,['A','C']]
"""
          A         C
a -0.529735 -1.067299
b -2.230089 -1.798575
c  0.685852  0.333387
d  1.061853  0.131853
e  0.990459  0.189966
f  0.057314 -0.370055
g  0.453960 -0.624419
h  0.666668 -0.433971
"""
# 注意，loc行参数是index
df.loc[['a','b','f','h'],['A','C']]
"""
          A         C
a -1.959731  0.720956
b  1.318976  0.199987
f -1.117735 -0.181116
h -0.147029  0.027369
"""
df.loc['a'] > 0
"""
A    False
B     True
C    False
D     True
Name: a, dtype: bool
"""
# 2 .iloc()
# iloc使用纯整数索引
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
df.iloc[:4]
"""
          A         B         C         D
0  0.277146  0.274234  0.860555 -1.312323
1 -1.064776  2.082030  0.695930  2.409340
2  0.033953 -1.155217  0.113045 -0.028330
3  0.241075 -2.156415  0.939586 -1.670171
"""
df.iloc[1:5, 2:4]
"""
          C         D
1  0.893615  0.659946
2  0.869331 -1.443731
3 -0.483688 -1.167312
4  1.566395 -1.292206
"""
df.iloc[1:3, :]
"""
          A         B         C         D
1 -0.133711  0.081257 -0.031869  0.009109
2  0.895576 -0.513450 -0.048573  0.698965
"""
# 3 .ix()
# ix可以使用标签和整数的混合运算
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
df.ix[:,'A']
"""
0    1.539915
1    1.359477
2    0.239694
3    0.563254
4    2.123950
5    0.341554
6   -0.075717
7   -0.606742
Name: A, dtype: float64
"""
```

### Pandas分组聚合

> 任何分组(*`groupby`*)操作都涉及原始对象的以下操作之一。它们是 ：
>
> - 分割对象
> - 应用一个函数
> - 结合的结果
>
> 在许多情况下，我们将数据分成多个集合，并在每个子集上应用一些函数。在应用函数中，可以执行以下操作 ：
>
> - *聚合* - 计算汇总统计
> - *转换* - 执行一些特定于组的操作
> - *过滤* - 在某些情况下丢弃数据
>
> `Pandas`对象可以分成任何对象。有多种方式来拆分对象
>
> - *`obj.groupby(‘key’)`*
> - *`obj.groupby([‘key1’,’key2’])`*
> - *`obj.groupby(key,axis=1)`*

```python
# 首先生成一个DataFrame
import pandas as pd

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
df 
"""
  Points  Rank    Team  Year
0      876     1  Riders  2014
1      789     2  Riders  2015
2      863     2  Devils  2014
3      673     3  Devils  2015
4      741     3   Kings  2014
5      812     4   kings  2015
6      756     1   Kings  2016
7      788     1   Kings  2017
8      694     2  Riders  2016
9      701     4  Royals  2014
10     804     1  Royals  2015
11     690     2  Riders  2017
"""
# 1.分组
df.groupby('Team')
# <pandas.core.groupby.DataFrameGroupBy object at 0x00000245D60AD518>
# 2.分组内容
df.groupby('Team').groups
"""
{
'Devils': Int64Index([2, 3], dtype='int64'), 
'Kings': Int64Index([4, 6, 7], dtype='int64'), 
'Riders': Int64Index([0, 1, 8, 11], dtype='int64'), 
'Royals': Int64Index([9, 10], dtype='int64'), 
'kings': Int64Index([5], dtype='int64')
}
"""
# 3.多键分组
df.groupby(['Team','Year']).groups
"""
{
('Devils', 2014): Int64Index([2], dtype='int64'), 
('Devils', 2015): Int64Index([3], dtype='int64'), 
('Kings', 2014): Int64Index([4], dtype='int64'),
('Kings', 2016): Int64Index([6], dtype='int64'),
('Kings', 2017): Int64Index([7], dtype='int64'), 
('Riders', 2014): Int64Index([0], dtype='int64'), 
('Riders', 2015): Int64Index([1], dtype='int64'), 
('Riders', 2016): Int64Index([8], dtype='int64'), 
('Riders', 2017): Int64Index([11], dtype='int64'),
('Royals', 2014): Int64Index([9], dtype='int64'), 
('Royals', 2015): Int64Index([10], dtype='int64'), 
('kings', 2015): Int64Index([5], dtype='int64')
}
"""
# 4.迭代输出分组
grouped = df.groupby('Year')

for name,group in grouped:
    print (name)
    print (group)
"""
2014
   Points  Rank    Team  Year
0     876     1  Riders  2014
2     863     2  Devils  2014
4     741     3   Kings  2014
9     701     4  Royals  2014
2015
    Points  Rank    Team  Year
1      789     2  Riders  2015
3      673     3  Devils  2015
5      812     4   kings  2015
10     804     1  Royals  2015
2016
   Points  Rank    Team  Year
6     756     1   Kings  2016
8     694     2  Riders  2016
2017
    Points  Rank    Team  Year
7      788     1   Kings  2017
11     690     2  Riders  2017
"""
# 5.获取分组
grouped = df.groupby("Year")
grouped.get_group(2014)
"""
   Points  Rank    Team  Year
0     876     1  Riders  2014
2     863     2  Devils  2014
4     741     3   Kings  2014
9     701     4  Royals  2014
"""
```

> 聚合函数为每个组返回单个聚合值。当创建了分组(*group by*)对象，就可以对分组数据执行多个聚合操作。
>
> 一个常用的方法是`agg`
>
> 分组或列上的转换返回索引大小与被分组的索引相同的对象。因此，转换应该返回与组块大小相同的结果。
>
> 转换的方法是`transform`
>
> 过滤根据定义的标准过滤数据并返回数据的子集。
>
> 过滤用的方法是`filter`

```python
# 首先生成一个DataFrame
import pandas as pd

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

# 1.聚合函数
grouped['Points'].agg(np.mean)
"""
Year
2014    795.25
2015    769.50
2016    725.00
2017    739.00
Name: Points, dtype: float64
"""
# 2.多个聚合函数
grouped = df.groupby('Team')
agg = grouped['Points'].agg([np.sum, np.mean, np.std])
"""
        sum        mean         std
Team                                
Devils  1536  768.000000  134.350288
Kings   2285  761.666667   24.006943
Riders  3049  762.250000   88.567771
Royals  1505  752.500000   72.831998
kings    812  812.000000         NaN
"""
# 3.转换函数
grouped = df.groupby('Team')
score = lambda x: (x - x.mean()) / x.std()*10
grouped.transform(score)
"""
       Points       Rank       Year
0   12.843272 -15.000000 -11.618950
1    3.020286   5.000000  -3.872983
2    7.071068  -7.071068  -7.071068
3   -7.071068   7.071068   7.071068
4   -8.608621  11.547005 -10.910895
5         NaN        NaN        NaN
6   -2.360428  -5.773503   2.182179
7   10.969049  -5.773503   8.728716
8   -7.705963   5.000000   3.872983
9   -7.071068   7.071068  -7.071068
10   7.071068  -7.071068   7.071068
11  -8.157595   5.000000  11.618950
"""
# 4.过滤函数
filter = df.groupby('Team').filter(lambda x: len(x) >= 3)
"""
   Points  Rank    Team  Year
0      876     1  Riders  2014
1      789     2  Riders  2015
4      741     3   Kings  2014
6      756     1   Kings  2016
7      788     1   Kings  2017
8      694     2  Riders  2016
11     690     2  Riders  2017
"""
```

### Pandas合并与级联

#### Pandas合并

> Pandas提供了一个单独的`merge()`函数来进行连接。

```python
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,left_index=False, right_index=False, sort=True)1
```

> - *left* - 一个DataFrame对象。
> - *right* - 另一个DataFrame对象。
> - *on* - 列(名称)连接，必须在左和右DataFrame对象中存在(找到)。
> - *left_on* - 左侧DataFrame中的列用作键，可以是列名或长度等于DataFrame长度的数组。
> - *right_on* - 来自右的DataFrame的列作为键，可以是列名或长度等于DataFrame长度的数组。
> - *left_index* - 如果为`True`，则使用左侧DataFrame中的索引(行标签)作为其连接键。 在具有MultiIndex(分层)的DataFrame的情况下，级别的数量必须与来自右DataFrame的连接键的数量相匹配。
> - *right_index* - 与右DataFrame的*left_index*具有相同的用法。
> - *how* - 它是*left*, *right*, *outer*以及*inner*之中的一个，默认为内*inner*。 下面将介绍每种方法的用法。
> - *sort* - 按照字典顺序通过连接键对结果DataFrame进行排序。默认为`True`，设置为`False`时，在很多情况下大大提高性能。

```python
# 连接实例
import pandas as pd
left = pd.DataFrame({
         'id':[1,2,3,4,5],
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame(
         {'id':[1,2,3,4,5],
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5']})
print (left)
print("========================================")
print (right)

"""
    Name  id subject_id
0    Alex   1       sub1
1     Amy   2       sub2
2   Allen   3       sub4
3   Alice   4       sub6
4  Ayoung   5       sub5
========================================
    Name  id subject_id
0  Billy   1       sub2
1  Brian   2       sub4
2   Bran   3       sub3
3  Bryce   4       sub6
4  Betty   5       sub5
"""
rs = pd.merge(left,right,on='id')
print(rs)
"""
   Name_x  id subject_id_x Name_y subject_id_y
0    Alex   1         sub1  Billy         sub2
1     Amy   2         sub2  Brian         sub4
2   Allen   3         sub4   Bran         sub3
3   Alice   4         sub6  Bryce         sub6
4  Ayoung   5         sub5  Betty         sub5
"""
```

`how`参数与`sql`:

| 合并方法    | SQL等效              | 描述       |
| ------- | ------------------ | -------- |
| `left`  | `LEFT OUTER JOIN`  | 使用左侧对象的键 |
| `right` | `RIGHT OUTER JOIN` | 使用右侧对象的键 |
| `outer` | `FULL OUTER JOIN`  | 使用键的联合   |
| `inner` | `INNER JOIN`       | 使用键的交集   |

```python
rs = pd.merge(left, right, on='subject_id', how='left')
"""
  Name_x  id_x subject_id Name_y  id_y
0    Alex     1       sub1    NaN   NaN
1     Amy     2       sub2  Billy   1.0
2   Allen     3       sub4  Brian   2.0
3   Alice     4       sub6  Bryce   4.0
4  Ayoung     5       sub5  Betty   5.0
"""
rs = pd.merge(left, right, on='subject_id', how='right')
"""
   Name_x  id_x subject_id Name_y  id_y
0     Amy   2.0       sub2  Billy     1
1   Allen   3.0       sub4  Brian     2
2   Alice   4.0       sub6  Bryce     4
3  Ayoung   5.0       sub5  Betty     5
4     NaN   NaN       sub3   Bran     3
"""
```

#### Pandas级联

```python
pd.concat(objs,axis=0,join='outer',join_axes=None,ignore_index=False)
```

> - *objs* - 这是Series，DataFrame或Panel对象的序列或映射。
> - *axis* - `{0，1，...}`，默认为`0`，这是连接的轴。
> - *join* - `{'inner', 'outer'}`，默认`inner`。如何处理其他轴上的索引。联合的外部和交叉的内部。
> - *ignore_index* − 布尔值，默认为`False`。如果指定为`True`，则不要使用连接轴上的索引值。结果轴将被标记为：`0，...，n-1`。
> - *join_axes* - 这是Index对象的列表。用于其他`(n-1)`轴的特定索引，而不是执行内部/外部集逻辑。

```python
import pandas as pd
one = pd.DataFrame({
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5'],
         'Marks_scored':[98,90,87,69,78]},
         index=[1,2,3,4,5])
two = pd.DataFrame({
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5'],
         'Marks_scored':[89,80,79,97,88]},
         index=[1,2,3,4,5])
rs = pd.concat([one,two])
print(rs)
"""
 Marks_scored    Name subject_id
1            98    Alex       sub1
2            90     Amy       sub2
3            87   Allen       sub4
4            69   Alice       sub6
5            78  Ayoung       sub5
1            89   Billy       sub2
2            80   Brian       sub4
3            79    Bran       sub3
4            97   Bryce       sub6
5            88   Betty       sub5
"""

rs = pd.concat([one,two],keys=['x','y'],ignore_index=True)
print(rs)
"""
   Marks_scored    Name subject_id
0            98    Alex       sub1
1            90     Amy       sub2
2            87   Allen       sub4
3            69   Alice       sub6
4            78  Ayoung       sub5
5            89   Billy       sub2
6            80   Brian       sub4
7            79    Bran       sub3
8            97   Bryce       sub6
9            88   Betty       sub5
  """
```

```python
# 附加连接 append
import pandas as pd
one = pd.DataFrame({
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5'],
         'Marks_scored':[98,90,87,69,78]},
         index=[1,2,3,4,5])
two = pd.DataFrame({
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5'],
         'Marks_scored':[89,80,79,97,88]},
         index=[1,2,3,4,5])
rs = one.append(two)
print(rs)
"""
   Marks_scored    Name subject_id
1            98    Alex       sub1
2            90     Amy       sub2
3            87   Allen       sub4
4            69   Alice       sub6
5            78  Ayoung       sub5
1            89   Billy       sub2
2            80   Brian       sub4
3            79    Bran       sub3
4            97   Bryce       sub6
5            88   Betty       sub5
"""
```

### Pandas时间工具

```python
# 时间工具实例
import pandas as pd

# 1.获取当前时间
pd.datetime.now()
# datetime.datetime(2019, 1, 10, 10, 56, 6, 388470)
# 2.创建时间戳
time = pd.Timestamp('2019-11-01')
time
# 2019-11-01 00:00:00
# 3.创建时间范围
time = pd.date_range("12:00", "23:59", freq="30min").time
time
"""
[datetime.time(12, 0) datetime.time(12, 30) datetime.time(13, 0)
 datetime.time(13, 30) datetime.time(14, 0) datetime.time(14, 30)
 datetime.time(15, 0) datetime.time(15, 30) datetime.time(16, 0)
 datetime.time(16, 30) datetime.time(17, 0) datetime.time(17, 30)
 datetime.time(18, 0) datetime.time(18, 30) datetime.time(19, 0)
 datetime.time(19, 30) datetime.time(20, 0) datetime.time(20, 30)
 datetime.time(21, 0) datetime.time(21, 30) datetime.time(22, 0)
 datetime.time(22, 30) datetime.time(23, 0) datetime.time(23, 30)]
 """
# 4.改变时间频率
"""
[datetime.time(12, 0) datetime.time(13, 0) datetime.time(14, 0)
 datetime.time(15, 0) datetime.time(16, 0) datetime.time(17, 0)
 datetime.time(18, 0) datetime.time(19, 0) datetime.time(20, 0)
 datetime.time(21, 0) datetime.time(22, 0) datetime.time(23, 0)]
 """
# 5.转换时间戳
# 要转换类似日期的对象(例如字符串，时代或混合)的序列或类似列表的对象，可以使用to_datetime函数。
# 当传递时将返回一个Series(具有相同的索引)，而类似列表被转换为DatetimeIndex
time = pd.to_datetime(pd.Series(['Jul 31, 2009','2019-10-10', None]))
time
"""
0   2009-07-31
1   2019-10-10
2          NaT
dtype: datetime64[ns]
"""
# 6.时间差函数 Timedelta
timediff = pd.Timedelta(6,unit='h')
# 0 days 06:00:00
timediff = pd.Timedelta(days=2)
# 2 days 00:00:00
```

### Pandas类别数据

> 通常实时的数据包括重复的文本列。例如：性别，国家和代码等特征总是重复的。这些是分类数据的例子。
>
> 分类变量只能采用有限的数量，而且通常是固定的数量。除了固定长度，分类数据可能有顺序，但不能执行数字操作。 分类是*`Pandas`*数据类型。
>
> 分类数据类型在以下情况下非常有用 :
>
> - 一个字符串变量，只包含几个不同的值。将这样的字符串变量转换为分类变量将会节省一些内存。
> - 变量的词汇顺序与逻辑顺序(`"one"`，`"two"`，`"three"`)不同。 通过转换为分类并指定类别上的顺序，排序和最小/最大将使用逻辑顺序，而不是词法顺序。
> - 作为其他python库的一个信号，这个列应该被当作一个分类变量(例如，使用合适的统计方法或`plot`类型)。
>
> 分类对象可以通过多种方式创建。
>
> - `pandas`对象创建中将`dtype`指定为`category`
> - `pd.Categorical`

```python
import pandas as pd
# 1.第一种，指定类型
s = pd.Series(["a","b","c","a"], dtype="category")
s
"""
0    a
1    b
2    c
3    a
dtype: category
Categories (3, object): [a, b, c]
"""
# 2.第二种，使用Categorical
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
cat
"""
[a, b, c, a, b, c]
Categories (3, object): [a, b, c]
"""
cat = pd.Categorical(['a','b','c','a','b','c','d'], ['c', 'b', 'a'],ordered=True)
cat
"""
[a, b, c, a, b, c, NaN]
Categories (3, object): [c < b < a]
"""
```

### Pandas IO工具

> Pandas I/O API是一套`pd.read_xxx()`，返回`Pandas`对象的顶级读取器函数。
>
> 例如：
>
> - `read_csv`
> - `read_excel`
> - `read_table`
> - `read_sql`
> - 等等

```python
pandas.read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer',names=None,
                index_col=None, usecols=None)
```

### Pandas自定义设置

> API由五个相关函数组成。它们分别是：
>
> - *`get_option()`*
> - *`set_option()`*
> - *`reset_option()`*
> - *`describe_option()`*
> - *`option_context()`*

| 编号   | 参数                          | 描述         |
| ---- | --------------------------- | ---------- |
| 1    | `display.max_rows`          | 要显示的最大行数   |
| 2    | `display.max_columns`       | 要显示的最大列数   |
| 3    | `display.expand_frame_repr` | 显示数据帧以拉伸页面 |
| 4    | `display.max_colwidth`      | 显示最大列宽     |
| 5    | `display.precision`         | 显示十进制数的精度  |

```python
import pandas as pd
# 1.get_option
print ("display.max_rows = ", pd.get_option("display.max_rows"))
# display.max_rows =  60
# 2.set_option
pd.set_option("display.max_rows",80)
print ("after set display.max_rows = ", pd.get_option("display.max_rows"))
# display.max_rows =  80

```

### Pandas运用实例

```python
import pandas as pd 
import numpy as np 

# 读入数据或者创建数据
# 1.读入csv文件
df = pd.DataFrame(pd.read_csv('city.csv',header=1))
# 2.或者使用pandas创建df
df = pd.DataFrame({"id":[1,2,3,4,5,6],
                   "date":pd.date_range('20130102', periods=6),
                   "city":['Beijing ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai', 'BEIJING '],
                   "age":[23,44,54,32,34,32],
                   "category":['100-A','100-B','110-A','110-C','210-A','130-F'],
                   "price":[1200,np.nan,2133,5433,np.nan,4432]},
                  columns =['id','date','city','category','age','price'])
df1=pd.DataFrame({"id":[1,2,3,4,5,6,7,8],
                  "gender"['male','female','male','female','male','female','male','female'],
                  "pay":['Y','N','Y','Y','N','Y','N','Y',],
                  "m-point":[10,12,20,40,40,40,30,20]})
# 基本信息查看
# 3.查看shape维度
# (6, 6)
# 4.查看info
df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 6 columns):
id          6 non-null int64
date        6 non-null datetime64[ns]
city        6 non-null object
category    6 non-null object
age         6 non-null int64
price       4 non-null float64
dtypes: datetime64[ns](1), float64(1), int64(2), object(2)
memory usage: 368.0+ bytes
"""
# 5.查看数据类型
df.dtypes
"""
id                   int64
date        datetime64[ns]
city                object
category            object
age                  int64
price              float64
dtype: object
"""
# 6.查看空值 
df.isnull()
"""
	id	date	city	category	age	price
0	False	False	False	False	False	False
1	False	False	False	False	False	True
2	False	False	False	False	False	False
3	False	False	False	False	False	False
4	False	False	False	False	False	True
5	False	False	False	False	False	False
"""
# 7.查看列名
df.columns
"""
Index(['id', 'date', 'city', 'category', 'age', 'price'], dtype='object')
"""
## 数据清洗
# 8.填充缺失值
df.fillna(value=0)
# 使用均值填充缺失值
df['prince'].fillna(df['prince'].mean())
# 9.大小写转换
df['city']=df['city'].str.lower()
# 10.更改列名
df.rename(columns={'category': 'category-size'})
# 11.更改数据类型
df['price'].astype('int')
# 12.去除重复值
df['city'].drop_duplicates(keep='last')
# 13.数据替换
df['city'].replace('sh', 'shanghai')
# 14.合并数据集
df_inner=pd.merge(df,df1,how='inner') # 匹配合并，交集
# 15.设置索引
df_inner.set_index('id')
# 16.排序
df_inner.sort_values(by=['age'])
# 17.根据条件处理新的一列
df_inner['group'] = np.where(df_inner['price'] > 3000,'high','low')
# 18.判断city列是否存在某个值
df_inner['city'].isin(['beijing'])
# 19.数据筛选
df_inner.loc[(df_inner['age'] > 25) & (df_inner['city'] == 'beijing'), ['id','city','age','category','gender']]
df_inner.query('city == ["beijing", "shanghai"]')
### 数据汇总统计
# 20.对city统计
df_inner.groupby('city').count()
# 21.根据city分组，并计算price的均值，总和，个数
df_inner.groupby('city')['price'].agg([len,np.sum, np.mean])
# 22.样本抽样,放回采样
df_inner.sample(n=6, replace=True)
# 23.计算两个字段的方差
df_inner['price'].cov(df_inner['m-point'])
# 24.计算所有字段之间的相关系数
df_inner.corr()
# 25.最后输出到csv
df_inner.to_csv('excel_to_python.csv')
```

### Pandas参考函数API

#### 构造函数

| 方法                                       | 描述    |
| ---------------------------------------- | ----- |
| DataFrame([data, index, columns, dtype, copy]) | 构造数据框 |

#### 属性和数据

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| Axes                                     | index: row labels；columns: column labels |
| DataFrame.as_matrix([columns])           | 转换为矩阵                                    |
| DataFrame.dtypes                         | 返回数据的类型                                  |
| DataFrame.ftypes                         | Return the ftypes (indication of sparse/dense and dtype) in this object. |
| DataFrame.get_dtype_counts()             | 返回数据框数据类型的个数                             |
| DataFrame.get_ftype_counts()             | Return the counts of ftypes in this object. |
| DataFrame.select_dtypes([include, exclude]) | 根据数据类型选取子数据框                             |
| DataFrame.values                         | Numpy的展示方式                               |
| DataFrame.axes                           | 返回横纵坐标的标签名                               |
| DataFrame.ndim                           | 返回数据框的纬度                                 |
| DataFrame.size                           | 返回数据框元素的个数                               |
| DataFrame.shape                          | 返回数据框的形状                                 |
| DataFrame.memory_usage([index, deep])    | Memory usage of DataFrame columns.       |

#### 类型转换

| 方法                                      | 描述          |
| --------------------------------------- | ----------- |
| DataFrame.astype(dtype[, copy, errors]) | 转换数据类型      |
| DataFrame.copy([deep])                  | 复制数据框       |
| DataFrame.isnull()                      | 以布尔的方式返回空值  |
| DataFrame.notnull()                     | 以布尔的方式返回非空值 |

#### 索引和迭代

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| DataFrame.head([n])                      | 返回前n行数据                                  |
| DataFrame.at                             | 快速标签常量访问器                                |
| DataFrame.iat                            | 快速整型常量访问器                                |
| DataFrame.loc                            | 标签定位                                     |
| DataFrame.iloc                           | 整型定位                                     |
| DataFrame.insert(loc, column, value[, …]) | 在特殊地点插入行                                 |
| DataFrame.**iter**()                     | Iterate over infor axis                  |
| DataFrame.iteritems()                    | 返回列名和序列的迭代器                              |
| DataFrame.iterrows()                     | 返回索引和序列的迭代器                              |
| DataFrame.itertuples([index, name])      | Iterate over DataFrame rows as namedtuples, with index value as first element of the tuple. |
| DataFrame.lookup(row_labels, col_labels) | Label-based “fancy indexing” function for DataFrame. |
| DataFrame.pop(item)                      | 返回删除的项目                                  |
| DataFrame.tail([n])                      | 返回最后n行                                   |
| DataFrame.xs(key[, axis, level, drop_level]) | Returns a cross-section (row(s) or column(s)) from the Series/DataFrame. |
| DataFrame.isin(values)                   | 是否包含数据框中的元素                              |
| DataFrame.where(cond[, other, inplace, …]) | 条件筛选                                     |
| DataFrame.mask(cond[, other, inplace, axis, …]) | Return an object of same shape as self and whose corresponding entries are from self where cond is False and otherwise are from other. |
| DataFrame.query(expr[, inplace])         | Query the columns of a frame with a boolean expression. |

#### 二元运算

| 方法                                       | 描述                                       |
| :--------------------------------------- | :--------------------------------------- |
| DataFrame.add(other[, axis, level, fill_value]) | 加法，元素指向                                  |
| DataFrame.sub(other[, axis, level, fill_value]) | 减法，元素指向                                  |
| DataFrame.mul(other[, axis, level, fill_value]) | 乘法，元素指向                                  |
| DataFrame.div(other[, axis, level, fill_value]) | 小数除法，元素指向                                |
| DataFrame.truediv(other[, axis, level, …]) | 真除法，元素指向                                 |
| DataFrame.floordiv(other[, axis, level, …]) | 向下取整除法，元素指向                              |
| DataFrame.mod(other[, axis, level, fill_value]) | 模运算，元素指向                                 |
| DataFrame.pow(other[, axis, level, fill_value]) | 幂运算，元素指向                                 |
| DataFrame.radd(other[, axis, level, fill_value]) | 右侧加法，元素指向                                |
| DataFrame.rsub(other[, axis, level, fill_value]) | 右侧减法，元素指向                                |
| DataFrame.rmul(other[, axis, level, fill_value]) | 右侧乘法，元素指向                                |
| DataFrame.rdiv(other[, axis, level, fill_value]) | 右侧小数除法，元素指向                              |
| DataFrame.rtruediv(other[, axis, level, …]) | 右侧真除法，元素指向                               |
| DataFrame.rfloordiv(other[, axis, level, …]) | 右侧向下取整除法，元素指向                            |
| DataFrame.rmod(other[, axis, level, fill_value]) | 右侧模运算，元素指向                               |
| DataFrame.rpow(other[, axis, level, fill_value]) | 右侧幂运算，元素指向                               |
| DataFrame.lt(other[, axis, level])       | 类似Array.lt                               |
| DataFrame.gt(other[, axis, level])       | 类似Array.gt                               |
| DataFrame.le(other[, axis, level])       | 类似Array.le                               |
| DataFrame.ge(other[, axis, level])       | 类似Array.ge                               |
| DataFrame.ne(other[, axis, level])       | 类似Array.ne                               |
| DataFrame.eq(other[, axis, level])       | 类似Array.eq                               |
| DataFrame.combine(other, func[, fill_value, …]) | Add two DataFrame objects and do not propagate NaN values, so if for a |
| DataFrame.combine_first(other)           | Combine two DataFrame objects and default to non-null values in frame calling the method. |

#### 函数应用&分组&窗口

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| [DataFrame.apply(func[, axis, broadcast, …\])](http://blog.csdn.net/claroja/article/details/74009232) | 应用函数                                     |
| DataFrame.applymap(func)                 | Apply a function to a DataFrame that is intended to operate elementwise, i.e. |
| DataFrame.aggregate(func[, axis])        | Aggregate using callable, string, dict, or list of string/callables |
| DataFrame.transform(func, *args, **kwargs) | Call function producing a like-indexed NDFrame |
| DataFrame.groupby([by, axis, level, …])  | 分组                                       |
| DataFrame.rolling(window[, min_periods, …]) | 滚动窗口                                     |
| DataFrame.expanding([min_periods, freq, …]) | 拓展窗口                                     |
| DataFrame.ewm([com, span, halflife, alpha, …]) | 指数权重窗口                                   |

#### 描述统计学

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| DataFrame.abs()                          | 返回绝对值                                    |
| DataFrame.all([axis, bool_only, skipna, level]) | Return whether all elements are True over requested axis |
| DataFrame.any([axis, bool_only, skipna, level]) | Return whether any element is True over requested axis |
| DataFrame.clip([lower, upper, axis])     | Trim values at input threshold(s).       |
| DataFrame.clip_lower(threshold[, axis])  | Return copy of the input with values below given value(s) truncated. |
| DataFrame.clip_upper(threshold[, axis])  | Return copy of input with values above given value(s) truncated. |
| DataFrame.corr([method, min_periods])    | 返回本数据框成对列的相关性系数                          |
| DataFrame.corrwith(other[, axis, drop])  | 返回不同数据框的相关性                              |
| DataFrame.count([axis, level, numeric_only]) | 返回非空元素的个数                                |
| DataFrame.cov([min_periods])             | 计算协方差                                    |
| DataFrame.cummax([axis, skipna])         | Return cumulative max over requested axis. |
| DataFrame.cummin([axis, skipna])         | Return cumulative minimum over requested axis. |
| DataFrame.cumprod([axis, skipna])        | 返回累积                                     |
| DataFrame.cumsum([axis, skipna])         | 返回累和                                     |
| DataFrame.describe([percentiles, include, …]) | 整体描述数据框                                  |
| DataFrame.diff([periods, axis])          | 1st discrete difference of object        |
| DataFrame.eval(expr[, inplace])          | Evaluate an expression in the context of the calling DataFrame instance. |
| DataFrame.kurt([axis, skipna, level, …]) | 返回无偏峰度Fisher’s (kurtosis of normal == 0.0). |
| DataFrame.mad([axis, skipna, level])     | 返回偏差                                     |
| DataFrame.max([axis, skipna, level, …])  | 返回最大值                                    |
| DataFrame.mean([axis, skipna, level, …]) | 返回均值                                     |
| DataFrame.median([axis, skipna, level, …]) | 返回中位数                                    |
| DataFrame.min([axis, skipna, level, …])  | 返回最小值                                    |
| DataFrame.mode([axis, numeric_only])     | 返回众数                                     |
| DataFrame.pct_change([periods, fill_method, …]) | 返回百分比变化                                  |
| DataFrame.prod([axis, skipna, level, …]) | 返回连乘积                                    |
| DataFrame.quantile([q, axis, numeric_only, …]) | 返回分位数                                    |
| DataFrame.rank([axis, method, numeric_only, …]) | 返回数字的排序                                  |
| DataFrame.round([decimals])              | Round a DataFrame to a variable number of decimal places. |
| DataFrame.sem([axis, skipna, level, ddof, …]) | 返回无偏标准误                                  |
| DataFrame.skew([axis, skipna, level, …]) | 返回无偏偏度                                   |
| DataFrame.sum([axis, skipna, level, …])  | 求和                                       |
| DataFrame.std([axis, skipna, level, ddof, …]) | 返回标准误差                                   |
| DataFrame.var([axis, skipna, level, ddof, …]) | 返回无偏误差                                   |

#### 从新索引&选取&标签操作

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| DataFrame.add_prefix(prefix)             | 添加前缀                                     |
| DataFrame.add_suffix(suffix)             | 添加后缀                                     |
| DataFrame.align(other[, join, axis, level, …]) | Align two object on their axes with the  |
| DataFrame.drop(labels[, axis, level, …]) | 返回删除的列                                   |
| [DataFrame.drop_duplicates([subset, keep, …\])](http://blog.csdn.net/claroja/article/details/76577793) | Return DataFrame with duplicate rows removed, optionally only |
| [DataFrame.duplicated([subset, keep\])](http://blog.csdn.net/claroja/article/details/76577793) | Return boolean Series denoting duplicate rows, optionally only |
| DataFrame.equals(other)                  | 两个数据框是否相同                                |
| DataFrame.filter([items, like, regex, axis]) | 过滤特定的子数据框                                |
| DataFrame.first(offset)                  | Convenience method for subsetting initial periods of time series data based on a date offset. |
| DataFrame.head([n])                      | 返回前n行                                    |
| DataFrame.idxmax([axis, skipna])         | Return index of first occurrence of maximum over requested axis. |
| DataFrame.idxmin([axis, skipna])         | Return index of first occurrence of minimum over requested axis. |
| DataFrame.last(offset)                   | Convenience method for subsetting final periods of time series data based on a date offset. |
| [DataFrame.reindex([index, columns\])](http://blog.csdn.net/claroja/article/details/72930594) | Conform DataFrame to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index. |
| [DataFrame.reindex_axis(labels[, axis, …\])](http://blog.csdn.net/claroja/article/details/72930594) | Conform input object to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index. |
| [DataFrame.reindex_like(other[, method, …\])](http://blog.csdn.net/claroja/article/details/72930594) | Return an object with matching indices to myself. |
| [DataFrame.rename([index, columns\])](http://blog.csdn.net/claroja/article/details/72930594) | Alter axes input function or functions.  |
| DataFrame.rename_axis(mapper[, axis, copy, …]) | Alter index and / or columns using input function or functions. |
| [DataFrame.reset_index([level, drop, …\])](http://blog.csdn.net/claroja/article/details/72930594) | For DataFrame with multi-level index, return new DataFrame with labeling information in the columns under the index names, defaulting to ‘level_0’, ‘level_1’, etc. |
| DataFrame.sample([n, frac, replace, …])  | 返回随机抽样                                   |
| DataFrame.select(crit[, axis])           | Return data corresponding to axis labels matching criteria |
| [DataFrame.set_index(keys[, drop, append, …\])](http://blog.csdn.net/claroja/article/details/72930594) | Set the DataFrame index (row labels) using one or more existing columns. |
| DataFrame.tail([n])                      | 返回最后几行                                   |
| DataFrame.take(indices[, axis, convert, is_copy]) | Analogous to ndarray.take                |
| DataFrame.truncate([before, after, axis, copy]) | Truncates a sorted NDFrame before and/or after some particular index value. |

#### 处理缺失值

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| [DataFrame.dropna([axis, how, thresh, …\])](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna) | Return object with labels on given axis omitted where alternately any |
| [DataFrame.fillna([value, method, axis, …\])](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna) | 填充空值                                     |
| [DataFrame.replace([to_replace, value, …\])](http://pandas.pydata.org/pandas-docs/stable/missing_data.html#replacing-generic-values) | Replace values given in ‘to_replace’ with ‘value’. |

#### 从新定型&排序&转变形态

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| DataFrame.pivot([index, columns, values]) | Reshape data (produce a “pivot” table) based on column values. |
| DataFrame.reorder_levels(order[, axis])  | Rearrange index levels using input order. |
| [DataFrame.sort_values(by[, axis, ascending, …\])](http://blog.csdn.net/claroja/article/details/73882340) | Sort by the values along either axis     |
| [DataFrame.sort_index([axis, level, …\])](http://blog.csdn.net/claroja/article/details/73882340) | Sort object by labels (along an axis)    |
| [DataFrame.nlargest(n, columns[, keep\])](http://blog.csdn.net/claroja/article/details/73882340) | Get the rows of a DataFrame sorted by the n largest values of columns. |
| [DataFrame.nsmallest(n, columns[, keep\])](http://blog.csdn.net/claroja/article/details/73882340) | Get the rows of a DataFrame sorted by the n smallest values of columns. |
| DataFrame.swaplevel([i, j, axis])        | Swap levels i and j in a MultiIndex on a particular axis |
| DataFrame.stack([level, dropna])         | Pivot a level of the (possibly hierarchical) column labels, returning a DataFrame (or Series in the case of an object with a single level of column labels) having a hierarchical index with a new inner-most level of row labels. |
| DataFrame.unstack([level, fill_value])   | Pivot a level of the (necessarily hierarchical) index labels, returning a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels. |
| DataFrame.melt([id_vars, value_vars, …]) | “Unpivots” a DataFrame from wide format to long format, optionally |
| DataFrame.T                              | Transpose index and columns              |
| DataFrame.to_panel()                     | Transform long (stacked) format (DataFrame) into wide (3D, Panel) format. |
| DataFrame.to_xarray()                    | Return an xarray object from the pandas object. |
| DataFrame.transpose(*args, **kwargs)     | Transpose index and columns              |

#### Combining& joining&merging

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| [DataFrame.append(other[, ignore_index, …\])](http://blog.csdn.net/claroja/article/details/72884998) | 追加数据                                     |
| DataFrame.assign(**kwargs)               | Assign new columns to a DataFrame, returning a new object (a copy) with all the original columns in addition to the new ones. |
| DataFrame.join(other[, on, how, lsuffix, …]) | Join columns with other DataFrame either on index or on a key column. |
| DataFrame.merge(right[, how, on, left_on, …]) | Merge DataFrame objects by performing a database-style join operation by columns or indexes. |
| DataFrame.update(other[, join, overwrite, …]) | Modify DataFrame in place using non-NA values from passed DataFrame. |

#### 时间序列

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| DataFrame.asfreq(freq[, method, how, …]) | 将时间序列转换为特定的频次                            |
| DataFrame.asof(where[, subset])          | The last row without any NaN is taken (or the last row without |
| DataFrame.shift([periods, freq, axis])   | Shift index by desired number of periods with an optional time freq |
| DataFrame.first_valid_index()            | Return label for first non-NA/null value |
| DataFrame.last_valid_index()             | Return label for last non-NA/null value  |
| DataFrame.resample(rule[, how, axis, …]) | Convenience method for frequency conversion and resampling of time series. |
| DataFrame.to_period([freq, axis, copy])  | Convert DataFrame from DatetimeIndex to PeriodIndex with desired |
| DataFrame.to_timestamp([freq, how, axis, copy]) | Cast to DatetimeIndex of timestamps, at beginning of period |
| DataFrame.tz_convert(tz[, axis, level, copy]) | Convert tz-aware axis to target time zone. |
| DataFrame.tz_localize(tz[, axis, level, …]) | Localize tz-naive TimeSeries to target time zone. |

#### 作图

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| [DataFrame.plot([x, y, kind, ax, ….\])](http://blog.csdn.net/claroja/article/details/73872066) | DataFrame plotting accessor and method   |
| DataFrame.plot.area([x, y])              | 面积图Area plot                             |
| DataFrame.plot.bar([x, y])               | 垂直条形图Vertical bar plot                   |
| DataFrame.plot.barh([x, y])              | 水平条形图Horizontal bar plot                 |
| DataFrame.plot.box([by])                 | 箱图Boxplot                                |
| DataFrame.plot.density(**kwds)           | 核密度Kernel Density Estimate plot          |
| DataFrame.plot.hexbin(x, y[, C, …])      | Hexbin plot                              |
| DataFrame.plot.hist([by, bins])          | 直方图Histogram                             |
| DataFrame.plot.kde(**kwds)               | 核密度Kernel Density Estimate plot          |
| DataFrame.plot.line([x, y])              | 线图Line plot                              |
| DataFrame.plot.pie([y])                  | 饼图Pie chart                              |
| DataFrame.plot.scatter(x, y[, s, c])     | 散点图Scatter plot                          |
| DataFrame.boxplot([column, by, ax, …])   | Make a box plot from DataFrame column optionally grouped by some columns or |
| DataFrame.hist(data[, column, by, grid, …]) | Draw histogram of the DataFrame’s series using matplotlib / pylab. |

#### 转换为其他格式

| 方法                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| DataFrame.from_csv(path[, header, sep, …]) | Read CSV file (DEPRECATED, please use pandas.read_csv() instead). |
| DataFrame.from_dict(data[, orient, dtype]) | Construct DataFrame from dict of array-like or dicts |
| DataFrame.from_items(items[, columns, orient]) | Convert (key, value) pairs to DataFrame. |
| DataFrame.from_records(data[, index, …]) | Convert structured or record ndarray to DataFrame |
| DataFrame.info([verbose, buf, max_cols, …]) | Concise summary of a DataFrame.          |
| DataFrame.to_pickle(path[, compression, …]) | Pickle (serialize) object to input file path. |
| DataFrame.to_csv([path_or_buf, sep, na_rep, …]) | Write DataFrame to a comma-separated values (csv) file |
| DataFrame.to_hdf(path_or_buf, key, **kwargs) | Write the contained data to an HDF5 file using HDFStore. |
| DataFrame.to_sql(name, con[, flavor, …]) | Write records stored in a DataFrame to a SQL database. |
| DataFrame.to_dict([orient, into])        | Convert DataFrame to dictionary.         |
| DataFrame.to_excel(excel_writer[, …])    | Write DataFrame to an excel sheet        |
| DataFrame.to_json([path_or_buf, orient, …]) | Convert the object to a JSON string.     |
| DataFrame.to_html([buf, columns, col_space, …]) | Render a DataFrame as an HTML table.     |
| DataFrame.to_feather(fname)              | write out the binary feather-format for DataFrames |
| DataFrame.to_latex([buf, columns, …])    | Render an object to a tabular environment table. |
| DataFrame.to_stata(fname[, convert_dates, …]) | A class for writing Stata binary dta files from array-like objects |
| DataFrame.to_msgpack([path_or_buf, encoding]) | msgpack (serialize) object to input file path |
| DataFrame.to_gbq(destination_table, project_id) | Write a DataFrame to a Google BigQuery table. |
| DataFrame.to_records([index, convert_datetime64]) | Convert DataFrame to record array.       |
| DataFrame.to_sparse([fill_value, kind])  | Convert to SparseDataFrame               |
| DataFrame.to_dense()                     | Return dense representation of NDFrame (as opposed to sparse) |
| DataFrame.to_string([buf, columns, …])   | Render a DataFrame to a console-friendly tabular output. |
| DataFrame.to_clipboard([excel, sep])     | Attempt to write text representation of object to the system clipboard This can be pasted into Excel, for example. |

### 参考链接

> - [csdn 基本函数](https://blog.csdn.net/HHTNAN/article/details/80080240)
> - [易百教程](https://www.yiibai.com/pandas/python_pandas_quick_start.html)
