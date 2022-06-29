# 剑指Offer之二叉搜索树的第k个结点


### 题目描述：

> 给定一棵二叉搜索树，请找出其中的第k小的结点。例如, （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

<!--more-->

### 解题思路：

> 按照树的中序遍历，然后找到第k小的结点
>
> 时间复杂度：$O(n)$, 空间复杂度$O(1)$.

```C++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    int count = 0;
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
       if(!pRoot) return NULL;
        
       if(pRoot)
        {
            TreeNode *p = KthNode(pRoot->left,k);
            if(p) return p;
           
            count++;
            if(count == k) return pRoot;
           
            p = KthNode(pRoot->right,k);
            if(p) return p;
        } 
        
       return NULL;
    }
};
```




