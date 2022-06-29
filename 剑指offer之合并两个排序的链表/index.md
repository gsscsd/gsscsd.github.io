# 剑指Offer之合并两个排序的链表


### 题目描述：

> 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

<!--more-->

### 解题思路：

> 时间复杂度：$O(n)$,空间复杂度：$O(1)$.

```C++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    // 合并有序链表
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        // 如果1为NULL，返回2
        if(!pHead1) return pHead2;
        // 如果2为NULL，返回1
        if(!pHead2) return pHead1;
        
        ListNode *p = pHead1;
        ListNode *q = pHead2;
        
        // 新建一个头节点
        ListNode *temp =  new ListNode(0);
        // 设置新的节点去遍历
        ListNode *pN = temp;
        
        // 循环遍历两个链表
        while(p && q)
        {
            // 比较大小，小的加载到新链表的后面
            if(p->val <= q->val)
            {
                pN -> next = p;
                p = p->next;
            }
            else
            {
                pN -> next = q;
                q = q->next;
            }
            pN = pN -> next;
        }
        // 如果p不为NULL，将所有的p加载到新链表
        if(p)
        {
            pN -> next = p;
        }
        // 如果q不为NULL，将所有的q加载到新链表
        if(q)
        {
            pN -> next = q;
        }
        return temp -> next;
    }
};
```


