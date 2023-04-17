# 剑指Offer之把数组排成最小的数


### 题目描述：

> 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

<!--more-->

### 解题思路：

> 时间复杂度: $O(nlogn)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    static bool com(string a,string b)
    {
        return (a+b) < (b+a);
    }
    string PrintMinNumber(vector<int> numbers) {
        string s = "";
        
        vector<string> vec;
        
        for(int i = 0; i < numbers.size(); i++)
        {
            vec.push_back(to_string(numbers[i]));
        }
        
        sort(vec.begin(),vec.end(),com);
        for(auto str : vec)
        {
            s += str;
        }
        return s;
    }
};
```


