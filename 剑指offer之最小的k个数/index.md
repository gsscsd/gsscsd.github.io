# 剑指Offer之最小的K个数


### 题目描述：

> 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

<!--more-->

### 解题思路：

#### 思路一：

> 时间复杂度: $O(nlogn)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(k > input.size()) return vector<int>();
        
        sort(input.begin(),input.end());
        return vector<int>(input.begin(),input.begin() + k);
    }
};
```

### 思路二：

> 冒泡排序 每次找最大或者最小的先排好，两个指针都指向前头
>
> 时间复杂度: $O(n^2)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(k > input.size()) return vector<int>();
        for(int i = 0; i < input.size(); i++)
        {
            int temp = 0;
            for(int j = i + 1; j < input.size(); j++)
            {
                if(input[i] > input[j])
                {
                    temp = input[i];
                    input[i] = input[j];
                    input[j] = temp;
                }
            }
        }
        
        return vector<int>(input.begin(), input.begin() + k);
    }
};
```

### 思路三：

> 快速排序  三个指针，两个指向前头，一个指向后面，然后找到一个数，数的一边比他小，另一边比他大
>
> 时间复杂度: $O(nlogn)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    
    void qsort(vector<int> &input, int low, int high)
    {
        if(low >= high)
        {
            return;
        }
        int k = input[low];
        int i = low;
        int j = high;
        
        while(i < j)
        {
            while(i < j && input[j] >= k) --j;
            input[i] = input[j];
            while(i < j && input[i] <= k) ++i;
            input[j] = input[i];
        }
        input[i] = k;
        qsort(input,low,i-1);
        qsort(input,i+1,high);
    }

    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(k > input.size()) return vector<int>();
        int i = 0;
        int j = input.size() - 1;
     
        qsort(input, i, j);
        
        
        return vector<int>(input.begin(), input.begin() + k);
    }
};
```


