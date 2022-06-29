# 剑指Offer之用两个栈实现队列


### 题目描述：

>用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型

<!--more-->

### 解题思路：

> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
class Solution
{
// 解题思路：
// 两个栈，s2加入新数据
// s1用来颠倒数据
public:
    // push 函数
    void push(int node) {
        // 首先，将s2中的数据出栈放到s1中
        while(!s2.empty())
        {
            s1.push(s2.top());
            s2.pop();
        }
        // 将新数据放到s2中
        s2.push(node);
        // 如果s1中的数据不为空，将数据放出栈放到s2中
        while(!s1.empty())
        {
            s2.push(s1.top());
            s1.pop();
        }
    }
	// 出栈数据
    int pop() {
        int x;
        x = s2.top();
        s2.pop();
        return x;
    }

private:
    stack<int> s1;
    stack<int> s2;
};
```


