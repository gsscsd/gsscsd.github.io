# 剑指Offer之字符流中第一个不重复的字符


### 题目描述：

 请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

输出描述：

```C++
 如果当前字符流没有存在出现一次的字符，返回#字符。
```

<!--more-->

### 解题思路：

时间复杂度：$O(n)$, 空间复杂度：$O(n)$.

```C++
class Solution
{
private:
    queue<char q;
    unsigned cnt[128] = {0};  // 保存每个字符出现的次数
public:
  //Insert one char from stringstream
    void Insert(char ch)
    {
        ++cnt[ch - '\0'];   //计算每个字符出现的次数
        q.push(ch);  //保存每个字符
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        while(!q.empty() && cnt[q.front()] = 2)
            q.pop();
        if(q.empty()) return '#';
        return q.front();
    }

};
```


