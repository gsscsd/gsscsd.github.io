# 剑指Offer之左旋转字符串


### 题目描述：

> 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

<!--more-->

### 解题思路一：

> 时间复杂度：$O(n)$,空间复杂度：$O(n)$.

```C++
class Solution {
public:
    string LeftRotateString(string str, int n) {
        string s1 = "";
        string s2 = "";
        
        for(int i = 0; i < n; i++)
        {
            s1 += str[i];
        }
        
        for(int j = n; j < str.length(); j++)
        {
            s2 += str[j];
        }
        
        return s2 + s1;
    }
};
```

### 解题思路二：

> 原理：YX = (XTY T)T
>
> 时间复杂度：$O(n)$,空间复杂度：$O(1)$.

```C++
class Solution {
public:
    string LeftRotateString(string str, int n)
    {
        int nLength = str.size();
        if(!str.empty() && n <= nLength)
        {
            if(n >= 0 && n <= nLength)
            {
                int pFirstStart = 0;
                int pFirstEnd = n - 1;
                int pSecondStart = n;
                int pSecondEnd = nLength - 1;
 
                // 翻转字符串的前面n个字符
                reverse(str, pFirstStart, pFirstEnd);
                // 翻转字符串的后面部分
                reverse(str, pSecondStart, pSecondEnd);
                // 翻转整个字符串
                reverse(str, pFirstStart, pSecondEnd);
            }
        }
        return str;
 
    }
    void reverse(string &str, int begin, int end)
    {
        while(begin < end)
        {
            char tmp = str[begin];
            str[begin] = str[end];
            str[end] = tmp;
            begin++;
            end--;
        }
    }
};
```


