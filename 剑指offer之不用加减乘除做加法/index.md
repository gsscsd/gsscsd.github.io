# 剑指Offer之不用加减乘除


### 题目描述

> 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

<!--more-->

### 解题思路

> 加法运算

```C++
class Solution {
public:
    int Add(int num1, int num2)
    { 
        return (num2 == 0)? num1:Add(num1^num2, (num1 & num2) << 1);
    }
};
```

> 减法运算

```C++
#include <iostream>
using namespace std;


int add(int num1, int num2)
{
	return (num2 == 0)? num1:add(num1^num2, (num1 & num2) << 1);
}

//求n的相反数
//~：按位取反
//add：加法操作，末位加一
int negative(int n)
{
	return add(~n, 1);
}

int subtraction(int n1, int n2)
{
     //加上被减数的相反数
	return add(n1, negative(n2));
}

int main()
{
	int sub = 0;
	sub = subtraction(-1, 2);	
	cout << sub;
	return 0;
}

```


