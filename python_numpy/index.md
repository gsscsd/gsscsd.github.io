# Numpy


> Numpy 里的数据结构和Python不同

<!--more-->

### Numpy Array只存一个Type

```python
# Numpy Array只存一个Type
python_list = ['string', 3, 3.2]
print(python_list)

import numpy as np
np_list1 = np.array([1, 2, 3, 4.0])
np_list2 = np.array([1, 2, 3], dtype = 'int')
print(np_list1)
np_list2[0] = 2.33333
print("2.33333 is trunated to %d" %np_list2[0])

# 输出如下所示：
"""
['string', 3, 3.2]
[1. 2. 3. 4.]
2.33333 is trunated to 2
"""
```

### 创建办法

```python
import numpy as np

zero_list = np.zeros(5, dtype = int)
one_matrix = np.ones((2, 2), dtype = float)
same_entry_matrix = np.full((1, 2), 2.333)
mean = 0
sd = 1
normal_number_matrix = np.random.normal(mean, sd, (3, 3))
print(zero_list)
print(one_matrix)
print(same_entry_matrix)
print(normal_number_matrix)

# 输出如下所示：
"""
[0 0 0 0 0]
[[1. 1.]
 [1. 1.]]
[[2.333 2.333]]
[[-0.16545138 -0.310813   -1.08602781]  # 随机的
 [ 1.99551174  0.19482657 -0.30166058]
 [ 0.60243205 -1.3135839   1.87655078]]
"""

start = 0
end = 10
step = 2
print(type(np.arange(start, end, step)) )

# 输出如下所示：
"""
<class 'numpy.ndarray'>
"""


```


