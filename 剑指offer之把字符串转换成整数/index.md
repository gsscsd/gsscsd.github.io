# 剑指Offer之把字符串转换成整数


### 题目描述：

 将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。

 ##### 输入描述:
输入一个字符串,包括数字字母符号,可以为空


##### 输出描述:

```C++
 如果是合法的数值表达则返回该数字，否则返回0
 ```

 示例1

##### 输入

 ```C++
 +2147483647
     1a33
 ```

##### 输出

 ```C++
 2147483647
     0
 ```

<!--more-->

### 解题思路：

 时间复杂度: $O(n)$,  空间复杂度: $O(1)$.

```C++
class Solution {
public:
    int StrToInt(string str) {
        if(str.length() == 0) return 0;
        
        int temp  = 1;
        int n = 0;
        
        // 处理正负号
        if(str[0] == '+') n = 1;
        if(str[0] == '-') 
        {
            temp = -1;
            n = 1;
        }
        // 按照每取一位相乘
        int sum = 0;
        for(int i = n; i < str.length(); i++)
        {
            if( str[i] < '0'  ||   str[i]  '9') return 0;
                
            sum = sum * 10 + str[i] - '0';
        }
        
        return temp * sum;
    }
};
```


