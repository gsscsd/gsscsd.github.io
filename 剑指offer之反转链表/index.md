# 剑指Offer之反转链表


### 题目描述：

> 输入一个链表，反转链表后，输出新链表的表头。

<!--more-->

### 解题思路一：

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
    ListNode* ReverseList(ListNode* pHead) {
        if(!pHead || !pHead->next) return pHead;
        
        ListNode *p = pHead;
        pHead = pHead->next;
        p->next = NULL;
        
        while(pHead->next)
        {
            ListNode *q = pHead;
            pHead = pHead->next;
            q->next = p;
            p = q;
        }
        
        pHead->next = p;
        return pHead;
    }
};
```

### 解题思路二：

> 递归方法:
>
> 递归的方法其实是非常巧的，它利用递归走到链表的末端，然后再更新每一个node的next 值 ，实现链表的反转。而newhead 的值没有发生改变，为该链表的最后一个结点，所以，反转后，我们可以得到新链表的head。
>
> 注意关于链表问题的常见注意点的思考：
>
> 1、如果输入的头结点是 NULL，或者整个链表只有一个结点的时候
>
> 2、链表断裂的考虑
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
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        //如果链表为空或者链表中只有一个元素
        if(pHead==NULL||pHead->next==NULL) return pHead;
         
        //先反转后面的链表，走到链表的末端结点
        ListNode* pReverseNode=ReverseList(pHead->next);
         
        //再将当前节点设置为后面节点的后续节点
        pHead->next->next=pHead;
        pHead->next=NULL;
         
        return pReverseNode;
         
    }
};
```


