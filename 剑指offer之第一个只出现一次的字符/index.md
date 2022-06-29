# 剑指Offer之第一个只出现一次的字符


### 题目描述：

> 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.

<!--more-->

### 解题思路一：

> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        map<char, int> map;
        
        for(int i = 0; i < str.length(); i++)
        {
            if(!map[str[i]]) map[str[i]] = 1;
            else map[str[i]]++;
        }
        
         for(int i = 0; i < str.length(); i++)
         {
             if(map[str[i]] == 1) return i;
         }
        
        return -1;
    }
};
```

### 解题思路二：

> find()的应用  （rfind() 类似，只是从反向查找）
>
> 找到 -- 返回 第一个字符的索引
>
> 没找到--返回   string::npos
>
> 时间复杂度: $O(n)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
    
         for(size_t i = 0; i < str.length(); i++)
         {
             if(str.find(str[i]) == str.rfind(str[i])) 
                 return i;
         }
        
        return -1;
    }
};
```


