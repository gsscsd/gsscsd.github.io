# 剑指Offer之求1+2+3+...+n


### 题目描述：

> 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

<!--more-->

### 解题思路：

>时间复杂度: $O(n)$， 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    int Sum_Solution(int n) {
        int res = n;
        res && (res += Sum_Solution(--n));   //与操作，前面为假，后面就不执行
        return res;
    }
};
```


