# 剑指Offer之矩形覆盖


### 题目描述：

> 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

<!--more-->

### 解题思路：

> 时间复杂度：$O(n)$, 空间复杂度：$O(n)$.

```C++
class Solution {
public:
    int rectCover(int number) {
        vector<int> vec(number + 1);
        vec[0] = 0;
        vec[1] = 1;
        vec[2] = 2;
        
        for(int i = 3; i <= number; i++)
            vec[i] = vec[i - 1] + vec[i - 2];
        
        return vec[number];
    }
};
```


