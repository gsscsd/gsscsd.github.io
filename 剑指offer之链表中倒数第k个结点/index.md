# 剑指Offer之链表中倒数第k个结点


### 题目描述：

> 输入一个链表，输出该链表中倒数第k个结点。

<!--more-->

### 解题思路：

>时间复杂度：$O(n)$, 空间复杂度：$O(1)$.

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
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        ListNode *p = pListHead;
        ListNode *q = pListHead;
        int i = 0;
        for( ;p != NULL;i++)
        {
            if(i >= k) q = q -> next;
            p = p -> next;
        }
        
        return i < k?NULL:q;
    }
};
```

### 解题思路二：

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
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        ListNode *p = pListHead;
        ListNode *q = pListHead;
        while(p && k > 0 )
        {
            p = p -> next;
            k--;
        }
        
        while(p)
        {
            p = p -> next;
            q = q -> next;
        }
        return k > 0 ? NULL:q;
    }
};
```


