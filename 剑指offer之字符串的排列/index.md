# 剑指Offer之数组中的逆序对


### 题目描述:

> 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
>
> 输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。

<!--more-->

### 解题思路一：

> 时间复杂度：$O(nlogn)$, 空间复杂度：$O(n)$.

```C++
class Solution {
public:
    vector<string> Permutation(string str) {
        vector<string> res;
        if(str.length() == 0) return res;
        // 先排序
        sort(str.begin(),str.end(),[](char a,char b){return a < b;});
        // 将第一个排列加入到数组中
        res.push_back(str);
        // 下一个全排列
        // next_permutation 获取下一个全排列
        while(next_permutation(str.begin(),str.end()))
        {i
            res.push_back(str);
        }
        return res;
    }
};
```

### 解题思路二：

> 思想：
>
> 1，3， 5， 7， 6， 4， 2， 1
>
> 第一步： 从后往前找到第一个比后面数小的,记做sw1.      1, 2, sw1, 7, 6, 4, 2, 1
>
> 第二步： 从后往前找到第一个比sw1数大的,记做sw2.       1, 2, sw1 5, 7, sw2 6, 4, 2, 1
>
> 第三步：交换sw1和 sw2                                                    1, 2, sw1 6, 7, sw2 5 4, 2, 1
>
> 第四步：sw1之后的数进行排序                                         1, 2, sw1 6,1, 2,  4, 5, 7
>
> 时间复杂度：$O(nlogn)$, 空间复杂度：$O(n)$.

```C++
class Solution {
public:
    vector<string> Permutation(string str) {
        vector<string> res;
        if(str.length() == 0) return res;
        // 第三个是一个lambda表达式
        sort(str.begin(),str.end(),[](char a,char b){return a < b;});
        res.push_back(str);
        while(findALL(str))
        {
            res.push_back(str);
        }
        return res;
    }
    // 自定义的下一个全排列
    bool findALL(string &str)
    {
        // 定义两个位置指针
        int sw1 = -1,sw2 = -1;
        // 从后向前寻找到第一个：str[i] > str[i + 1] 位置指针
        for(int i = str.length() - 2;i >= 0;i--)
        {
            if(str[i ] < str[i + 1])
            {
                sw1 = i ;
                break;
            }
        }
        // 如果没有找到，说明没有下一个全排列
        if(sw1 == -1) return false;
        // 从后向前寻找，找到第一个大于str[sw1]的位置指针
        for(int j = str.length() - 1;j >= sw1;j--)
        {
             if(str[j] > str[sw1])
             {
                 sw2 = j;
                 break;
             }
         }
       // 交换str的两个位置指针
       swap(str[sw1],str[sw2]);
       // 将sw1后面的数全排列
       sort(str.begin() + sw1 + 1,str.end());
       return true;
        
    }   
};
   
```


