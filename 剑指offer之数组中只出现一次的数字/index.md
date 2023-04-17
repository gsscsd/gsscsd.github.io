# 剑指Offer之数组中只出现一次的数字


### 题目描述：

> 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

<!--more-->

### 解题思路：

> 时间复杂度：$O(n)$,空间复杂度：$O(1)$.

```C++
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        sort(data.begin(), data.end());
        
        int flag = 0;
        
        for(int i = 0; i < data.size(); )
        {
            if(data[i] != data[i + 1])
            {
                if(flag == 0)
                {
                    *num1 = data[i];
                    flag = 1;
                    i++;
                    continue;
                }
                if(flag == 1)
                {
                    *num2 = data[i];
                    break;
                }
            }
            else
                 i = i + 2;
        }
    }
};
```


