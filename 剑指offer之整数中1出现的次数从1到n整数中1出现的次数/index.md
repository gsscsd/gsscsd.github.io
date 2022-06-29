# 剑指Offer之整数中1出现的次数（从1到n整数中1出现的次数）


### 题目描述：

> 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

<!--more-->

### 解题思路一：

> 时间复杂度：$O(n^2)$, 空间复杂度：$O(1)$.

```C++
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n)
    {
        int count = 0;
         for(int i = 1; i <= n; i++)
         {
             string ss = to_string(i);

             for(int j = 0; j < ss.length(); j++)
             {
                 if(ss[j] == '1')
                     count++;
             }
         }
        
        return count;
    }
};
```

### 解题思路二：

> 时间复杂度：$O(n^2)$, 空间复杂度：$O(1)$.

```C++
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n)
    {
        int count=0;
        if(n<1) return 0;
        for(int i=1;i<=n;++i)
        {
            int temp=i;
            while(temp)
            {
                if(temp%10==1)
                    ++count;
                temp/=10;
            }
        }
        return count;
    }
};
```

### 
