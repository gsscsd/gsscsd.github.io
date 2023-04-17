# 剑指Offer之两个链表的第一个公共结点


### 题目描述：

> 输入两个链表，找出它们的第一个公共结点。

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
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        ListNode *ff;
        ListNode *h1 = pHead1, *h2 = pHead2;
        
        int count1 = 0, count2 = 0, count3 = 0;
        while(h1)
        {
            count1++;
            h1 = h1->next;
        }
        while(h2) 
        {
            count2++;
            h2 = h2->next;
        }
        
        if(count1 >= count2)
        {
            count3 = count1 - count2;
            while(count3)
            {
                pHead1 = pHead1->next;
                count3--;
            }
        }
        if(count1 < count2)
        {
            count3 = count2 - count1;
            while(count3)
            {
                pHead2 = pHead2->next;
                count3--;
            }
        }
        
        while(pHead1 &&  pHead2)
        {
            if(pHead1 == pHead2)
            {
                
                ff = pHead2;
                break;
            }
            pHead1 = pHead1->next;
            pHead2 = pHead2->next;
        }
        
        return ff;
    }
};
```

### 解题思路二：

>时间复杂度：$O(n)$,空间复杂度：$O(1)$.

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
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        ListNode *p1 = pHead1;
        ListNode *p2 = pHead2;
        while(p1!=p2)
        {
            p1 = (p1==NULL ? pHead2 : p1->next);
            p2 = (p2==NULL ? pHead1 : p2->next);
        }
        return p1;
    }
};
```


