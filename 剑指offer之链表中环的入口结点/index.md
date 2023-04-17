# 剑指Offer之链表中环的入口结点


### 题目描述：

> 给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

<!--more-->

### 解题思路：

> 两个指针一个fast、一个slow同时从一个链表的头部出发, fast一次走2步，slow一次走一步，如果该链表有环，两个指针必然在环内相遇,此时只需要把其中的一个指针重新指向链表头部，另一个不变（还在环内），这次两个指针一次走一步，相遇的地方就是入口节点。 
>
> 时间复杂度：$O(n^2)$, 空间复杂度：$O(1)$.

```C++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        ListNode* p = pHead;
        ListNode* q = pHead;
        if(pHead == NULL)
            return NULL;
        while(q->next != NULL && q->next->next != NULL){
            p = p->next;
            q = q->next->next;
            if(p == q)
            {
                q = pHead;
                while(p != q)
                {
                    p = p->next;
                    q = q->next;
                }
                return p;
            }
        }
        return NULL;
    }
};
```

>时间复杂度：$O(n^2)$, 空间复杂度：$O(1)$.

```C++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if (!pHead || pHead->next == NULL) return NULL;
        
        ListNode *h1 = pHead;
        ListNode *h2 = pHead;
        ListNode *h3 = pHead;
        int flag = 0;
        
        while(h2)
        {
            h1 = h1->next;
            h2 = h2->next->next;
            if(h1 == h2)
            {
                flag = 1;
                break;
            }
        }
        while(flag)
        {
            if(h1 == h3)
                return h1;
            h1 = h1->next;
            h3 = h3->next;
         }
        
        return NULL;

    }
};
```


