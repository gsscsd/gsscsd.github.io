# 剑指Offer之滑动窗口的最大值


### 题目描述：

> 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

<!--more-->

### 解题思路：

> 时间复杂度: $O(n^2)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        vector<int> vec;
        if(size == 0 || size > num.size()) return vec;
        
        int s = num.size();
        int m = s - size + 1;
        
        for(int i = 0; i < num.size(); i++)
        {
            int temp = num[i];
            int j = i;
            while( (j < size + i) && m)
            {
                if(temp < num[j]) temp = num[j];
                j++;
            }
            m--;
            vec.push_back(temp);
            if(!m) break;
        }
        
        return vec;
    }
};
```


