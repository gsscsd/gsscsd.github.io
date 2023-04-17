# 剑指Offer之丑数


### 题目描述：

> 把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

<!--more-->

### 解题思路：

> 时间复杂度：$O(n)$, 空间复杂度：$O(n)$.

```C++
class Solution {
public:
int GetUglyNumber_Solution(int index) {
    	 // 0-6的丑数分别为0-6
        if (index < 7)return index;
        vector<int> res(index);
        res[0] = 1;
    	//  t2，t3，t5分别为三个队列的指针
        int t2 = 0, t3 = 0, t5 = 0, i;
        for (i = 1; i < index; ++i)
        {
            // 选出三个队列头最小的数
            res[i] = min(res[t2] * 2, min(res[t3] * 3, res[t5] * 5));
             // 这三个if有可能进入一个或者多个，进入多个是三个队列头最小的数有多个的情况
            if (res[i] == res[t2] * 2)t2++;
            if (res[i] == res[t3] * 3)t3++;
            if (res[i] == res[t5] * 5)t5++;
        }
        return res[index - 1];
    }
};
```


