# C++之定义声明初始化与赋值


### <center >前言

> 在学习C++以及其他编程语言的时候，经常会碰到一个问题，那就是变量定义、声明、初始化与赋值都有什么样的区别呢？

<!--more-->

变量的定义、声明、初始化和赋值这四个概念在C++中是很容易区分，所以，从C++入手来学习并且这四个概念，在其他的编程语言里面，也不会迷惑了。

### 1.变量的定义与初始化

> 变量的定义：顾名思义，变量的定义就是指在使用这个变量之前，先对它进行**定义并申请存储空间**，在变量定义的时候，还可以对它进行初始化。
>
> **PS：变量只能定义一次，也只能初始化一次。**

比如说看下面的代码：

```C++
int sum; // 这一行代码就是对sum这个变量的定义，此时sum并没有被赋值和初始化，但是已经被分配存储空间，sum可能会根据不同的编译器来决定是否被默认的去初始化，有可能sum里面的数是垃圾数字。
int count = 0; // 这一行代码是对count变量的定义以及初始化为0;
```

### 2.变量的声明

> 变量的声明：变量的声明仅仅只是**规定了变量的类型和名字**，并不分配存储空间。
>
> PS:一般我们也可以把定义当做声明，**变量可以被多次声明，但是只能被定义一次。**

在C++中的声明以及定义：

```C++
extern int i; // 使用关键字extern，我们就可以声明一个变量了，注意此处的i一定会在其他地方被定义，此处这是声明变量i
int j; // 声明并且定义了变量j
```

### 3.变量的赋值与初始化

> 在前面说到，**初始化只能是在定义的时候进行，并且变量也只能初始化一次**，而赋值可以进行多次。

```C++
int i = 0; // 定义声明变量i，并且将i初始化为0
i = 9;  // i赋值为9
i = 6;  // i赋值为6
```

### 在java语言中的定义、声明、初始化和赋值

> 在java中，变量只有两种，一种是基本类型，一种是对象。**基本类型变量的声明和定义（初始化）是同时产生的；而对于对象来说，声明和定义是分开的。** 
>
> 在java中，变量的类型和各种修饰符的搭配有很多，因此，变量的初始化的过程也很复杂，在这里，只是记录一下，基本类型和其他引用类型在类成员和成员函数里面的时候的默认初始化规则。
>
> 总结一句话：如果基本类型是在成员函数里面，那么，**一定要初始化，否则会报错**。其他情况，编译器会自动根据类型来初始化。

```java
/*************************************************************************
	> File Name: demo.java
	> Author: 
	> Mail: 
	> Created Time: 二 12/ 4 13:59:23 2018
 ************************************************************************/

public class demo
{
    public static void main(String[] args)
    {
        test t = new test();
        t.test();
    }
}
class test
{
    int i;
    int j = 9;
    public void test()
    {
        int k;
        int l = 7;
        System.out.println(i);
        System.out.println(j);
        System.out.println(k);
        System.out.println(l);
    }
}

// 程序编译错误，原因在于k没有去初始化。
// i变量默认为0
```

### 在Python中的定义、声明、初始化和赋值

> Python是解释型的动态语言，因此，在使用变量的时候，可以不用声明和定义，如果想要使用一个变量，直接去初始化一个变量或者对某个变量赋值。

```python
i = 9 //定义初始化一个变量i
j = i ** 2 // 定义初始化变量j
```


