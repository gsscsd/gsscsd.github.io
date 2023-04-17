# 剑指Offer之数值的整数次方


### 题目描述：

> 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

<!--more-->

### 解题思路一：

```C++
class Solution {
public:
    double Power(double base, int exponent) {
        
       base = pow(base, exponent);
        
       return base;
    }
};
```

### 解题思路二：

> 时间复杂度：$O(n)$, 空间复杂度：$O(1)$.

```C++
class Solution {
public:
    double Power(double base, int exponent) {
        if(exponent == 0) return 1.0;
        // 判断exponent是正数还是负数
        int sign = exponent < 0 ? 1:0;
        exponent = abs(exponent);
        int res = base;
        
        while(--exponent)
        {
            base *= res;
        }
        // 如果是负数，要求倒数
        if(sign) base = 1 / base;
        
        return base;
    }
};
```


