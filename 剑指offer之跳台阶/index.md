# 剑指Offer之跳台阶


### 题目描述：

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

<!--more-->

### 解题思路：

> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    int jumpFloor(int number) {
        vector<int> vec(number + 1);
        
        vec[0] = 0;
        vec[1] = 1;
        vec[2] = 2;
        
        for(int i = 3; i <= number; ++i)
        {
            vec[i] = vec[i - 1] + vec[i - 2];
        }
        
        return vec[number];
    }
};
```


