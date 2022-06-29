# Python基础语法


### Python基础语法

####  Boolean and its Opeations

```python
# Boolean and its Operations
print(type(True))
print(type(False))
print(type(None))

# 输出如下所示：
"""
<class 'bool'>
<class 'bool'>
<class 'NoneType'>
"""
```

<!--more-->

#### And  Or

```python
# and or
B = True
A = False

print(B and True)
print(A or B)
print(not B)
print(not A)
print(B and (not A))

# 输出如下所示：
"""
True
True
False
True
True
"""
```

#### Expressions can evaluate to a boolean value

```python
# Expressions can evaluate to a boolean value
print(3 >= 2)
print(2 >= 2)
print(A is A)
print(A is B)
print(3 != 2)

# 输出如下所示：
"""
True
True
True
False
True
"""
```

#### Integer and floats

```python
# Integer and floats
print(type(3))
print(type(3.0))

# 输出如下所示：
"""
<class 'int'>
<class 'float'>
"""
```

#### Numerical operations

```python
# Numerical operations
import math

print(3 / 2)
print(5 // 2)
print(3 % 2)
print(float(3))
print(pow(2, 3))
print(math.ceil(3.4))
print(math.floor(3.4))
print(round(3.1415926, 2))

# 输出如下所示：
"""
1.5
2
1
3.0
8
4
3
3.14
"""
```

#### String

> - strings are useful because they are immutable（它们不可变）

```python
# String
Introduction = 'Today is 2019-01-21'
print(Introduction)
print(str(3))
print(int('3'))
print(str("b'AS"))

# String Method
A = 'abc'
print(A.capitalize())
print(A.upper())
print(A.find('b'))
print(A.find('d'))

# String formatting
s = 'My name is Tom'
name = 'Tomas'
age = 25
score = 98.532421
print('hello, my name is %s and I am %d years old\n \ I scored %f on my midterm'%(name, age, score))
print('hello, my name is %s and I am %d years old\n \
I scored %.2f on my midterm'%(name, age, score))

# 输出如下所示：
"""
Today is 2019-01-21
3
3
b'AS

Abc
ABC
1
-1

hello, my name is Tomas and I am 25 years old
 \ I scored 98.532421 on my midterm
hello, my name is Tomas and I am 25 years old
 I scored 98.53 on my midterm
"""
```

#### List

> - Mutable（可变的）
> - Iterable
> - Sequence Type

```python
# List 
# Create and Modify list
A = []
A.append(1)
A
# List index and Slicing
# length = n, start from 0 and end at (n-1)
A = [1, 2, 3, 4, 5, 6]
print(len(A))
print(A[0])
A[0] = 100
print(A[0])
A[1:4]
print(A[1:4])  # 不会输出索引4的数

# What exactly is "iterable"?
for item in A[2:5]:
    print(item)
A = [1, 2]
A.append('2')
print(A)
A[0] = 's'
print(A)
print(list('abc') )

# 输出如下所示：
"""
6
1
100
[2, 3, 4]

3
4
5
[1, 2, '2']
['s', 2, '2']
['a', 'b', 'c']
"""
```

####  Method and Function

```python
# Method
A = list((2, 1, 3))
print(A)
A.sort(reverse = True) # A的值发生了改变
print(A)
# Function
print(sorted(A))  # A的值没有改变
print(A)

# 输出如下所示：
"""
[2, 1, 3]
[3, 2, 1]
[1, 2, 3]
[3, 2, 1]
"""
```

#### Tuple

> - Similar to list 和list类似
> - But it is immutable(hashable) 但它是不可变的 support for hash(), a built-in function
> - Can have repeated value可以有重复的元素
> - Sequence Type

```python
# Tuple
# Create Tuple
A = tuple([1, 2, 3, 2])
print(A)
print(A[0])
print(A[2:4])
print(A[:4])
print(A[::])
print(A[:-1])
B = tuple([1, 2, 3, 4])
print(B[::-1])

A[0] = 's'
print(A)

C = tuple()
print(C)
C = 1,
print(C)
D = (1)
print(type(D))


# 输出如下所示：
"""
(1, 2, 3, 2)
1
(3, 2)
(1, 2, 3, 2)
(1, 2, 3, 2)
(1, 2, 3)
(4, 3, 2, 1)

A[0] = 's'
TypeError: 'tuple' object does not support item assignment

()
(1,)
<class 'int'>
"""
```

#### Set

> - Unordered 没有顺序
> - Contain hashable elements( so it is again immutable) 里面的元素必须是hashable
> - No repetition 没有重复
> - Not a sequence type

```python
# Set
A = set([1, 1, 2])
B = set(['Michal', 'Betty'])
print(A)
print(len(A))
print(B)
print(len(B))

C = set([1], [2])
print(C)

# Set Arithmetics
A = set([1, 2, 3])
B = set([3, 4, 5])
print('A intersects B')
print(A & B)
print('A unions B') 
print(A | B)
print('in A not in B')
print(A - B)
print('add 5 into A')
A.add(5)
print(A)
print('new A after adding 5')
print(A)
print('remove 2 in A')
A.remove(2)
print(A)
print('new A after reoving 2')
print(A)

# 输出如下所示：
"""
{1, 2}
2
{'Michal', 'Betty'}
2

C = set([1], [2])
TypeError: set expected at most 1 arguments, got 2

A intersects B
{3}
A unions B
{1, 2, 3, 4, 5}
in A not in B
{1, 2}
add 5 into A
{1, 2, 3, 5}
new A after adding 5
{1, 2, 3, 5}
remove 2 in A
{1, 3, 5}
new A after reoving 2
{1, 3, 5}
"""
```

####  Common operations on List, Set, Tuple

```python
# Use tuple for demonstration here
A = (1, 2, 3)
print(len(A))

for item in A:
    print(item)

print(1 in A)

# 输出如下所示：
"""
3
1
2
3
True
"""
```

#### Dictionary

> - Very fast for look up( O(1) compared to O(N) in list) 非常快
> - Mapping Type (think of "Map" as "Match") #### Keys
> - Arbitary values 任意值
> - Must be hashable(i.e. Immutable, will explain later) 必须是不变的
> - We may also call them "index"(pandas) 有时候也称为index #### Values
> - No restrictions, can be mutable 没有限制

```python
# Dictionary
def simple_function(a, b, c):
    print(a)
    print(b)
    print(c)

simple_function(1, 2, 3)
simple_function(c = 3, a = 1, b = 2)
simple_function(1, c = 3, b = 2)

# 输出如下所示：
"""
1
2
3
1
2
3
1
2
3
"""

#### Create List 创建list
midterm1 = {'Michael': 98, 'Betty': 96}
print(midterm1)

final = dict() # 一个空字典
print(final)

midterm2 = dict(Michael = 98, Betty = 96)
print(midterm2)

midterm3 = dict([('Michael', 98), ('Betty', 96)])
print(midterm3)

print(midterm1 == midterm2 == midterm3)

midterm4 = dict([('Michael', 98), ('Betty', 96), ('Michael', 90)])
print(midterm4)

midterm1['Michael'] = 97
print(midterm1)

for key in midterm1:
    print(key)

print(midterm1.keys())
print(midterm1.values())

# 输出如下所示：
{'Michael': 98, 'Betty': 96}
{}
{'Michael': 98, 'Betty': 96}
{'Michael': 98, 'Betty': 96}
True
{'Michael': 90, 'Betty': 96}
{'Michael': 97, 'Betty': 96}
Michael
Betty
dict_keys(['Michael', 'Betty'])
dict_values([97, 96])
```

#### Ranges

> - Start 开始
> - Stop 结束
> - Step 步的大小

```python
# Ranges
print(list(range(10)))
print(list(range(0, 10, 3)))
print(list(range(0, -10, -1)))
print(list(range(0)))
print(list(range(1, 0)))

r = range(10)
print(5 in r)
print(r[1])

# 输出如下所示：
"""
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[0, 3, 6, 9]
[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
[]
[]
True
1
"""
```

#### Hashable

> - map name strings to 15 integers
> - red = a collision
> - think of them as many buckets
> - theoretically more than one items in every single bucket, but we say the look up is O(1)

```python
# Hashable
print(hash('apple'))
print(hash(((1, -1, 0), (1, 0, 0), (1, 0, -1))))
print(hash(((1, 0, -1), (1, 0, 0), (1, -1, 0))))

# 输出如下所示：
-4061574024490551516
-697649482279922733
-697649482279922733
```

#### Loop and Conditionals

```python
# If Statement
name = 'Kesci'
if name == 'Kesci':
    print(name)

tool = 'Klab'
if tool != 'Kesci':
    print(tool)
    
# Use Good Grammar for Efficiency
# if if if
# if elif else
number = 10
def print_silly(x):
    if number == 10:
        print('checking the first one')
        print('equal to 10')
    if number <= 10:
        print('checking the second one')
        print('no less than 10')
    if number >= 10:
        print('checking the third one')
        print('no more than 10')

def print_clever(x):
    if number == 10:
        print('checking the first one')
        print('equal to 10')
    elif number <= 10:
        print('checking the second one')
        print('no less than 10')
    elif number >= 10:
        print('checking the third one')
        print('no more than 10')

print_silly(10)
print_clever(10)
    
 
# 输出如下所示：
"""
Kesci
Klab

checking the first one
equal to 10
checking the second one
no less than 10
checking the third one
no more than 10

checking the first one
equal to 10
"""

# For Loop
# When you know when to stop 当你知道你什么时候停止
count = 0
for number in range(10):
    count = count + number
    print(count)
    
# 输出如下所示：
"""
0
1
3
6
10
15
21
28
36
45
"""

# While loop
# Interchangeable with for loop 可以和for loop互换
# Must use this one if you don't know when to stop 当你不知道什么时候停下来
# Easy to have bug( forever looping) if you forget to modify your condition 容易有bug
unknown_condition = 100
my_number = 10
while my_number < unknown_condition:
    my_number = my_number + 10
    print(my_number)

print('I am out of while loop now')
print(my_number)

# 输出如下所示：
"""
20
30
40
50
60
70
80
90
100
I am out of while loop now
100
"""

# Nest your if statement in your loops:
# Use indentation
unknown_condition = 100
my_number = 0
while my_number < unknown_condition:
    if my_number % 2 == 0:
        my_number = my_number + 10
        print(my_number)
    else:
        my_number = my_number + 5
        print(my_number)

print('I am out of while loop now')
print('my_number is %d in the end'%my_number)

# 输出如下所示：
"""
10
20
30
40
50
60
70
80
90
100
I am out of while loop now
my_number is 100 in the end
"""
```

#### 2-D List, List Aliasing, Shallow or Deep Copy

> - Where we get most bugs
> - You think you get a new list, but you are getting the same one in reality **“牵一发动全身”** **“一改全改了”**
> - When you are copying a list

```python
A = [1,2]
B = A[:]
print(B)

print(B is A)

A = [[1,2,3],[4,5,6]]
B = A[:]
print(A)
print(B)

B[1][2] = 100
print(B)

print(A)

print(A is B)
print(A[0] is B[0])
print(A[1] is B[1])

# 输出如下所示：
"""
[1, 2]
False
[[1, 2, 3], [4, 5, 6]]
[[1, 2, 3], [4, 5, 6]]
[[1, 2, 3], [4, 5, 100]]
[[1, 2, 3], [4, 5, 100]]
False
True
True
"""

[[1, 2, 3], [4, 5, 100]]
[[1, 2, 3], [4, 5, 6]]
[[1, 2, 3], [4, 5, 100]]
# We call B a shallow copy of A, to make a deep copy,
# we need to use the copy.deepcopy function
import copy
A = [[1,2,3],[4,5,6]]
B = copy.copy(A)
B[1][2] = 100
print(A) # same as [:]

A = [[1,2,3],[4,5,6]]
B = copy.deepcopy(A)
B[1][2] = 100
print(A)
print(B)

# 输出如下所示：
"""
[[1, 2, 3], [4, 5, 100]]
[[1, 2, 3], [4, 5, 6]]
[[1, 2, 3], [4, 5, 100]]
"""
```


