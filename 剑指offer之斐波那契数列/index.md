# 剑指Offer之斐波那契数列


### 题目描述：

> 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
>
> n<=39

<!--more-->

### 解题思路一：

> 斐波那契数列（Fibonacci sequence），又称[黄金分割](https://baike.baidu.com/item/%E9%BB%84%E9%87%91%E5%88%86%E5%89%B2/115896)数列、因[数学家](https://baike.baidu.com/item/%E6%95%B0%E5%AD%A6%E5%AE%B6/1210991)列昂纳多·斐波那契（Leonardoda Fibonacci）以兔子繁殖为例子而引入，故又称为“[兔子数列](https://baike.baidu.com/item/%E5%85%94%E5%AD%90%E6%95%B0%E5%88%97/6849441)”，指的是这样一个数列：1、1、2、3、5、8、13、21、34、……在数学上，斐波纳契数列以如下被以递推的方法定义：F(1)=1，F(2)=1, F(n)=F(n-1)+F(n-2)（n>=3，n∈N*）, 这个数列从第3项开始，每一项都等于前两项之和。
>
> 时间复杂度：$O(n)$, 空间复杂度：$O(n)$.

```C++
class Solution {
public:
    int Fibonacci(int n) {
        vector<int> vec(n+1);
        
        vec[0] = 0;
        vec[1] = 1;
        vec[2] = 1;
        
        for(int i = 3; i <= n; i++)
            vec[i] = vec[i - 1] + vec[i - 2];
        
        return vec[n];

    }
};
```

### 解题思路二：

> 时间复杂度：$O(n)$, 空间复杂度：$O(1)$.

```C++
class Solution {
public:
    int Fibonacci(int n) {
        int first = 0;
        int second = 1;
         
        int result = n;
        for(int i = 2; i<=n; i++){
            result = first + second;
            first = second;
            second = result;
        }
        return result;
    }
};
```


