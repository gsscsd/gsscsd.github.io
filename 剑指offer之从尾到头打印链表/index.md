# 剑指Offer之从尾到头打印链表


### 题目描述：

> 输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

<!--more-->

### 解题思路：

> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> vec;
        
        if(head == NULL) return vec;
       
        while(head->next != NULL)
        {
            vec.insert(vec.begin(),head->val);
            head = head->next;
        }
        
        vec.insert(vec.begin(),head->val);
        return vec;   
    }
};
```


