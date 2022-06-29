# 剑指Offer之删除链表中重复的结点


### 题目描述：

> 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

<!--more-->

### 解题思路一:

> 非递归
>
> 时间复杂度：$O(n)$, 空间复杂度：$O(1)$.

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
class Solution
{
public:
    ListNode *deleteDuplication(ListNode *pHead)
    {
        if (!pHead) return NULL;
        if (!pHead->next) return pHead;

        ListNode *pre = new ListNode(0);
        pre->next = pHead;

        ListNode *p = pre;
        ListNode *q = pHead;

        while (q)
        {
            // 如果 q 和 q->next相同，那个把那个值记录下来，和后面的比较
            while (q != NULL && q->next != NULL && q->next->val == q->val)
            {
                int tmp = q->val;
                // 凡是相等的直接往后移
                while (q != NULL && q->val == tmp)
                    q = q->next;
            }
            // 直到找到不相同的，把q 给 p的next 
            p->next = q;
            // p 移向下一个
            p = p->next;
            if (q)  q = q->next;
        }

        return pre->next;
    }
};
```

### 解题思路二：

> 递归
>
> 时间复杂度:$O(n)$,空间复杂度：$O(1)$.

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
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if (pHead==NULL)
            return NULL;
        if (pHead!=NULL && pHead->next==NULL)
            return pHead;
                 
        ListNode* current;
         
        if ( pHead->next->val==pHead->val){
            current=pHead->next->next;
            while (current != NULL && current->val==pHead->val)
                current=current->next;
            return deleteDuplication(current);                     
        }
         
        else {
            current=pHead->next;
            pHead->next=deleteDuplication(current);
            return pHead;
        }    
    }
};
```


