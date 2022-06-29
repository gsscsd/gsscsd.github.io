# 剑指Offer之二维数组中的查找


### 题目描述：

> 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

<!--more-->

### 解题思路：

> 时间复杂度: $O(n)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int a = array.size();  //数组的行数
        int b = array[0].size();  //数组的列数
       
        for(int i = a - 1, j = 0; i >= 0 && j < b;)
        {
            if(target == array[i][j]) return true;
            if(target > array[i][j]) j++;
            if(target < array[i][j]) i--;
        }
        
       
        
        return false;
    }
};
```


