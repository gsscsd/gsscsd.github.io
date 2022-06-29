# 剑指Offer之包含min函数的栈


### 题目描述：

> 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

<!--more-->

### 解题思路：

```C++
class Solution {
public:
    void push(int value) {
        st.push(value);
        if(smin.empty())
            smin.push(value);
        if(smin.top()>value)
            smin.push(value);
    }
    void pop() {
        if(smin.top()==st.top())
            smin.pop();
        st.pop();
    }
    int top() {
        return st.top();
    }
    int min() {
        return smin.top();
    }
    private:
    stack<int> st;
    stack<int> smin;
};
```


