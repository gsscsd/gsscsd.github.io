# 剑指Offer之翻转单词顺序列


### 题目描述：

> 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

<!--more-->

### 解题思路一：

> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    string ReverseSentence(string str) {
        string s = "";
        stack<string> st;
        if(str.size() == 0 ) return s;
        
        string temp = "";
        
        for(int i = 0; i <= str.length(); i++)
        {
            if(str[i] == ' ' || str[i] == '\0')
            {
                st.push(temp);
                temp = "";
            }
            else temp += str[i];
        }
        
        while(!st.empty())
        {
            s += st.top();
            s += ' ';
            st.pop();
        }
        s.pop_back();
        
        return s;
    }
};
```

### 解题思路二：

> 时间复杂度: $O(n)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    string ReverseSentence(string str) {
        string res = "", tmp = "";
        for(unsigned int i = 0; i < str.size(); ++i){
            if(str[i] == ' ') res = " " + tmp + res, tmp = "";
            else tmp += str[i];
        }
        if(tmp.size()) res = tmp + res;
        return res;
    }
};
```


