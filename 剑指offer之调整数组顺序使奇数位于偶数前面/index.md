# 剑指Offer之调整数组顺序使奇数位于偶数前面


### 题目描述：

> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

<!--more-->

### 解题思路一：

> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        vector<int> vec1;
        vector<int> vec2;
        
        for(int i = 0; i < array.size(); i++)
        {
            if(array[i] % 2 == 0) vec2.push_back(array[i]);
            else vec1.push_back(array[i]);
        }
        for(int j = 0; j < vec1.size(); j++) array[j] =  vec1[j];
        for(int z = 0; z < vec2.size(); z++) array[ vec1.size()+ z] =  vec2[z];
    }
};
```

### 解题思路二：

> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        vector<int> vec;
        
        for(int i = 0; i < array.size(); i++)
        {
            if(array[i] % 2 != 0) vec.push_back(array[i]);
        }
        for(int j = 0; j < array.size(); j++) 
        {
            if(array[j] % 2 == 0) vec.push_back(array[j]);
        }
        
        array = vec;
    }
};
```


