# 剑指Offer之和为S的两个数字


### 题目描述：

>
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

###### 输出描述:

```C++
对应每个测试案例，输出两个数，小的先输出。
```

<!--more-->

### 解题思路：

#### 思路一：

暴力循环
>
时间复杂度: $O(n^2)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    vector<intFindNumbersWithSum(vector<intarray,int sum) {
        vector<intvec;
        int i = 0, j = 1;
        int temp = INT_MAX;
        
        for(int i = 0; i < array.size(); i++)
        {
            for(int j = i + 1; j < array.size(); j++)
            {
                 if(array[i] + array[j] == sum)
                 {
                    if(array[i] * array[j] < temp)
                    {
                        temp = array[i] * array[j];
                        vec.push_back(array[i]);
                        vec.push_back(array[j]);
                    }
                  }
            }
        }
            
        return vec;
    }
};
```

#### 思路二：

双指针
>
时间复杂度: $O(n)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    vector<intFindNumbersWithSum(vector<intarray,int sum) {
        vector<intvec;
        int i = 0;
        int j = array.size() - 1;
        int Sum = 0;
        
        while(i < j)
        {
            Sum = array[i] + array[j];
            if(Sum sum)
                j--;
            if(Sum < sum)
                i++;
            if(Sum == sum)
            {
                vec.push_back(array[i]);
                vec.push_back(array[j]);
                break;
            }
        }
        
        return vec;
    }
};
```


