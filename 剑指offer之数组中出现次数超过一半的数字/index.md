# 剑指Offer之数组中出现次数超过一半的数字


### 题目描述：

> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

<!--more-->

### 解题思路一：

> 时间复杂度：$O(n^2)$,空间复杂度：$O(1)$.

```C++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int temp = numbers.size() / 2;
        
        if(numbers.size() == 1) return numbers[0];
        
        for(int i = 0; i < numbers.size(); i++)
        {
            int count = 0;
            for(int j = i + 1; j < numbers.size(); j++)
            {
                if(numbers[i] == numbers[j])
                {
                    count++;
                    if(count >= temp)
                        return numbers[i];
                }
            }
        }
        
        return 0;
    }
};
```

### 解题思路二：

> 时间复杂度：$O(n)$,空间复杂度：$O(1)$.

```C++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int temp = numbers.size() / 2 ;
        
        if(numbers.size() == 1) return numbers[0];
        
        map<int,int> map;
        for(int i = 0; i < numbers.size(); i++)
        {
            if(!map[numbers[i]]) map[numbers[i]]=1;
            else map[numbers[i]]++;
            if( map[numbers[i]] > temp) return  numbers[i];
        }
        return 0;
    }
};
```


